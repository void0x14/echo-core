//! Deney C (uçtan uca): gerçek decode döngüsüne istasyon boru hattı ekler.
//! Her decode adımından sonra ağırlık havuzunun tamamı pipe'tan geçer (süre ayrı ölçülür).
//! "none" aynı döngüyü boru hattı olmadan koşar (taban).
//! Çalıştırma: zig-out/bin/deney-c-e2e <model.gguf> <none|single|station> <max_tokens> <p> <q> <tohum>
//! Çıktı: stderr'de tek satır JSON.

const std = @import("std");
const ports = @import("ports/inference.zig");
const stations = @import("stations.zig");

const BLK = stations.BLK;

fn threadCpuNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.THREAD_CPUTIME_ID, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args_it.deinit();
    var args = std.array_list.Managed([]const u8).init(gpa);
    defer args.deinit();
    while (args_it.next()) |arg| try args.append(arg);

    if (args.items.len != 7) {
        std.debug.print("kullanım: {s} <model.gguf> <none|single|station> <max_tokens> <p> <q> <tohum>\n", .{args.items[0]});
        return error.MissingArgument;
    }

    const cond = args.items[2];
    const use_pipe = !std.mem.eql(u8, cond, "none");
    const mode = if (use_pipe) (std.meta.stringToEnum(stations.Mode, cond) orelse return error.UnknownMode) else stations.Mode.single;
    const max_tokens = try std.fmt.parseInt(u32, args.items[3], 10);
    const p = try std.fmt.parseFloat(f64, args.items[4]);
    const q = try std.fmt.parseFloat(f64, args.items[5]);
    const seed = try std.fmt.parseInt(u64, args.items[6], 10);

    var port = try ports.InferencePort.init(args.items[1], gpa);
    defer port.deinit(gpa);
    port.reset();

    var ids = try port.tokenizer_.encode("");
    defer ids.deinit();
    const eng = &port.loader.engine_;
    _ = try eng.prefill(ids.items);

    const pool = eng.weight_pool;
    const chunks: u64 = pool.len / BLK;
    if (chunks == 0) return error.PoolTooSmall;

    var pipe = try stations.Pipe.init(gpa, .{ .mode = mode, .p = p, .q = q, .seed = seed });
    defer pipe.deinit(gpa);
    const scratch = try gpa.alloc(u8, BLK);
    const per_tok_ms = try gpa.alloc(f64, max_tokens);

    const io = std.Io.Threaded.global_single_threaded.io();
    var st = stations.Stats{};
    var decode_ns: u64 = 0;
    var decode_cpu_ns: u64 = 0;
    var pipe_ns: u64 = 0;
    var pipe_cpu_ns: u64 = 0;
    var pipe_samples: usize = 0;
    var seq: u64 = 0;
    var tokens: u32 = 0;

    while (tokens < max_tokens) : (tokens += 1) {
        const next_id = eng.greedyNextToken();
        if (next_id == port.tokenizer_.eos()) break;
        if (eng.kv_cache) |cache| {
            if (cache.seqLen() >= eng.config.max_seq_len) break;
        }

        const d0 = std.Io.Timestamp.now(io, .awake);
        const dc0 = threadCpuNs();
        _ = try eng.decodeStep(next_id);
        decode_cpu_ns += threadCpuNs() - dc0;
        decode_ns += @intCast(d0.untilNow(io, .awake).toNanoseconds());

        if (use_pipe) {
            const r0 = std.Io.Timestamp.now(io, .awake);
            const rc0 = threadCpuNs();
            var i: u64 = 0;
            while (i < chunks) : (i += 1) {
                _ = pipe.pass(seq + i, pool[@intCast(i * BLK) ..][0..BLK], scratch, &st);
            }
            const pass_cpu = threadCpuNs() - rc0;
            const pass_ns: u64 = @intCast(r0.untilNow(io, .awake).toNanoseconds());
            pipe_cpu_ns += pass_cpu;
            pipe_ns += pass_ns;
            per_tok_ms[pipe_samples] = @as(f64, @floatFromInt(pass_ns)) * 1e-6;
            pipe_samples += 1;
            seq += chunks;
        }
    }

    // Doğrulama (süre dışı): teslim edilen her blok, havuzdaki özgün baytla karşılaştırılır.
    var ver_correct: u64 = 0;
    var ver_wrong: u64 = 0;
    var ver_lost: u64 = 0;
    if (use_pipe) {
        const out = try gpa.alloc(u8, BLK);
        defer gpa.free(out);
        var vst = stations.Stats{};
        var i: u64 = 0;
        while (i < chunks) : (i += 1) {
            const x = pool[@intCast(i * BLK) ..][0..BLK];
            if (!pipe.pass(seq + i, x, out, &vst)) {
                ver_lost += 1;
                continue;
            }
            if (std.mem.eql(u8, x, out)) ver_correct += 1 else ver_wrong += 1;
        }
    }

    const dec_s = @as(f64, @floatFromInt(decode_ns)) * 1e-9;
    const pipe_s = @as(f64, @floatFromInt(pipe_ns)) * 1e-9;
    const n_tok: f64 = @floatFromInt(tokens);
    const tps_engine = n_tok / dec_s;
    const tps_total = n_tok / (dec_s + pipe_s);

    const samples = per_tok_ms[0..pipe_samples];
    var p50: f64 = 0;
    var p99: f64 = 0;
    if (pipe_samples > 0) {
        std.mem.sort(f64, samples, {}, std.sort.asc(f64));
        p50 = samples[(pipe_samples - 1) / 2];
        p99 = samples[((pipe_samples - 1) * 99) / 100];
    }

    std.debug.print(
        "{{\"cond\":\"{s}\",\"p\":{d},\"q\":{d},\"seed\":{d},\"tokens\":{d}," ++
            "\"decode_ms\":{d:.1},\"decode_cpu_ms\":{d:.1},\"tg_tps_engine\":{d:.3}," ++
            "\"pipe_ms\":{d:.1},\"pipe_cpu_ms\":{d:.1},\"pipe_ms_per_tok\":{d:.2},\"pipe_p50_ms\":{d:.2},\"pipe_p99_ms\":{d:.2}," ++
            "\"tg_tps_total\":{d:.3},\"chunks\":{d},\"pool_bytes\":{d}," ++
            "\"detected\":{d},\"retries\":{d},\"lost\":{d},\"crc_checks\":{d}," ++
            "\"ver_correct\":{d},\"ver_wrong\":{d},\"ver_lost\":{d}}}\n",
        .{
            cond, p, q, seed, tokens,
            @as(f64, @floatFromInt(decode_ns)) * 1e-6,
            @as(f64, @floatFromInt(decode_cpu_ns)) * 1e-6,
            tps_engine,
            @as(f64, @floatFromInt(pipe_ns)) * 1e-6,
            @as(f64, @floatFromInt(pipe_cpu_ns)) * 1e-6,
            if (pipe_samples > 0) @as(f64, @floatFromInt(pipe_ns)) * 1e-6 / @as(f64, @floatFromInt(pipe_samples)) else 0,
            p50,
            p99,
            tps_total,
            chunks,
            pool.len,
            st.detected,
            st.retries,
            st.lost,
            st.crc_checks,
            ver_correct,
            ver_wrong,
            ver_lost,
        },
    );
}
