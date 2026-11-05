//! Deney A (blok düzeyi): tek çekirdekte dolaşan blok ring'i, GGUF'taki gerçek tensör baytları üzerinde.
//! Çalıştırma: zig-out/bin/deney-a <model.gguf> <direct|checksum|recon> <blok> <p> <q> <tohum> <cpu>
//! Çıktı: stderr'de tek satır JSON.

const std = @import("std");
const gguf = @import("gguf");
const ring = @import("ring/ring.zig");

const BLK = ring.BLK;
const WIN = 1024;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args_it.deinit();
    var args = std.array_list.Managed([]const u8).init(gpa);
    defer args.deinit();
    while (args_it.next()) |arg| try args.append(arg);

    if (args.items.len != 8) {
        std.debug.print("kullanım: {s} <model.gguf> <direct|checksum|recon> <blok> <p> <q> <tohum> <cpu>\n", .{args.items[0]});
        return error.MissingArgument;
    }

    const mode = std.meta.stringToEnum(ring.Mode, args.items[2]) orelse return error.UnknownCondition;
    const n_blocks = try std.fmt.parseInt(u64, args.items[3], 10);
    const p = try std.fmt.parseFloat(f64, args.items[4]);
    const q = try std.fmt.parseFloat(f64, args.items[5]);
    const seed = try std.fmt.parseInt(u64, args.items[6], 10);
    const cpu = try std.fmt.parseInt(usize, args.items[7], 10);
    if (n_blocks == 0) return error.EmptyRun;

    var set = std.mem.zeroes(std.os.linux.cpu_set_t);
    set[cpu / @bitSizeOf(usize)] |= @as(usize, 1) << @as(u6, @intCast(cpu % @bitSizeOf(usize)));
    try std.os.linux.sched_setaffinity(0, &set);

    var reader = try gguf.Reader.openWithAllocator(args.items[1], gpa);
    defer reader.deinit();
    var name: ?[]const u8 = null;
    var best_size: u64 = 0;
    var it = reader.tensors.iterator();
    while (it.next()) |e| {
        if (std.mem.endsWith(u8, e.key_ptr.*, "ffn_down.weight") and e.value_ptr.size > best_size) {
            name = e.key_ptr.*;
            best_size = e.value_ptr.size;
        }
    }
    const tensor_name = name orelse return error.TensorNotFound;
    const weights = try reader.loadTensor(tensor_name);
    const chunks: u64 = weights.len / BLK;
    if (chunks == 0) return error.TensorTooSmall;

    var r = try ring.Ring.init(gpa, .{ .mode = mode, .p = p, .q = q, .seed = seed });
    defer r.deinit(gpa);
    const deliv = try gpa.alloc(u8, WIN * BLK);
    const lat = try gpa.alloc(u32, n_blocks);
    const done = try gpa.alloc(bool, n_blocks);

    const io = std.Io.Threaded.global_single_threaded.io();
    var st = ring.Stats{};
    var correct: u64 = 0;
    var wrong: u64 = 0;
    var wall_ns: u64 = 0;

    var w: u64 = 0;
    while (w < n_blocks) : (w += WIN) {
        const n: u64 = @min(@as(u64, WIN), n_blocks - w);
        const w0 = std.Io.Timestamp.now(io, .awake);

        var j: u64 = 0;
        while (j < n) : (j += 1) {
            const seq = w + j;
            const t0 = std.Io.Timestamp.now(io, .awake);
            const x = weights[@intCast((seq % chunks) * BLK) ..][0..BLK];
            const ok = r.passBlock(seq, x, deliv[@intCast(j * BLK) ..][0..BLK], &st);
            done[@intCast(seq)] = ok;
            const dt: u64 = @intCast(t0.untilNow(io, .awake).toNanoseconds());
            lat[@intCast(seq)] = @intCast(@min(dt, std.math.maxInt(u32)));
        }

        wall_ns += @intCast(w0.untilNow(io, .awake).toNanoseconds());

        // Oracle süre dışı: teslim edilen her blok özgün girdiyle karşılaştırılır.
        var jj: u64 = 0;
        while (jj < n) : (jj += 1) {
            const seq = w + jj;
            if (!done[@intCast(seq)]) continue;
            const x = weights[@intCast((seq % chunks) * BLK) ..][0..BLK];
            if (std.mem.eql(u8, x, deliv[@intCast(jj * BLK) ..][0..BLK])) correct += 1 else wrong += 1;
        }
    }

    std.mem.sort(u32, lat, {}, std.sort.asc(u32));
    const p50 = lat[@intCast((n_blocks - 1) / 2)];
    const p99 = lat[@intCast(((n_blocks - 1) * 99) / 100)];

    const wall_s = @as(f64, @floatFromInt(wall_ns)) * 1e-9;
    const blocks_per_s = @as(f64, @floatFromInt(n_blocks)) / wall_s;
    const useful_mb_s = @as(f64, @floatFromInt(correct)) * BLK / wall_s / 1e6;

    std.debug.print(
        "{{\"cond\":\"{s}\",\"blocks\":{d},\"p\":{d},\"q\":{d},\"seed\":{d},\"cpu\":{d},\"chunks\":{d},\"tensor\":\"{s}\"," ++
            "\"wall_s\":{d:.6},\"blocks_per_s\":{d:.1},\"useful_mb_s\":{d:.2},\"p50_ns\":{d},\"p99_ns\":{d}," ++
            "\"delivered\":{d},\"correct\":{d},\"wrong\":{d},\"lost\":{d},\"detected\":{d},\"retries\":{d}," ++
            "\"recovered\":{d},\"uncleaned\":{d}}}\n",
        .{
            @tagName(mode), n_blocks, p,            q,          seed,           cpu,
            chunks,         tensor_name,   wall_s,     blocks_per_s, useful_mb_s, p50,
            p99,            st.delivered,  correct,    wrong,      st.lost,        st.detected,
            st.retries,     st.recovered,  r.uncleaned(),
        },
    );
}
