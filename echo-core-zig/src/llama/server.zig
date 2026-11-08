//! OpenAI uyumlu HTTP sunucu (llama-server benzeri, tek istek sırası).
//! POST /v1/chat/completions, POST /v1/completions (stream: SSE), GET /v1/models, GET /health
const std = @import("std");
const cli = @import("cli.zig");
const sampler = @import("sampler.zig");
const sablon = @import("sablon.zig");

const ArrayList = std.array_list.Managed;
const Value = std.json.Value;

pub fn serve(s: anytype, host: []const u8, port: u16, model_name: []const u8) !void {
    s.kayit = .{};
    s.kayit_ids = ArrayList(u32).init(s.a);
    defer {
        s.kayit.?.deinit(s.a);
        s.kayit_ids.deinit();
    }
    const addr = try std.Io.net.IpAddress.parse(host, port);
    var srv = try addr.listen(s.io, .{ .reuse_address = true });
    defer srv.deinit(s.io);
    std.debug.print("echo-core sunucu: http://{s}:{d}  (POST /v1/chat/completions, /v1/completions; GET /v1/models, /health)\n", .{ host, port });
    while (true) {
        const conn = srv.accept(s.io) catch |e| {
            std.debug.print("accept hatası: {}\n", .{e});
            continue;
        };
        defer conn.close(s.io);
        handle(s, conn, model_name) catch |e| std.debug.print("istek hatası: {}\n", .{e});
    }
}

const Req = struct { method: []const u8, path: []const u8, body: []const u8 };

fn readReq(r: *std.Io.Reader, a: std.mem.Allocator) !Req {
    var req: Req = .{ .method = "", .path = "", .body = "" };
    const first = (try r.takeDelimiter('\n')) orelse return error.EndOfStream;
    var it = std.mem.tokenizeScalar(u8, std.mem.trimEnd(u8, first, "\r"), ' ');
    req.method = try a.dupe(u8, it.next() orelse return error.BadRequest);
    req.path = try a.dupe(u8, it.next() orelse return error.BadRequest);
    var clen: usize = 0;
    while (true) {
        const line = std.mem.trimEnd(u8, (try r.takeDelimiter('\n')) orelse return error.EndOfStream, "\r");
        if (line.len == 0) break;
        if (std.ascii.startsWithIgnoreCase(line, "content-length:"))
            clen = try std.fmt.parseInt(usize, std.mem.trim(u8, line[15..], " "), 10);
    }
    if (clen > 16 << 20) return error.BodyTooLarge;
    const body = try a.alloc(u8, clen);
    try r.readSliceAll(body);
    req.body = body;
    return req;
}

fn yanit(w: *std.Io.Writer, code: []const u8, ctype: []const u8, body: []const u8) !void {
    try w.print("HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", .{ code, ctype, body.len });
    try w.writeAll(body);
    try w.flush();
}

fn hata(w: *std.Io.Writer, code: []const u8, msg: []const u8, a: std.mem.Allocator) !void {
    const b = try std.json.Stringify.valueAlloc(a, .{ .@"error" = .{ .message = msg, .type = "invalid_request_error" } }, .{});
    try yanit(w, code, "application/json", b);
}

fn num(v: ?Value) ?f64 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}
fn str(v: ?Value) ?[]const u8 {
    const x = v orelse return null;
    return switch (x) {
        .string => |t| t,
        else => null,
    };
}

/// SSE alıcısı: her metin parçasını bir chunk olarak yollar.
const Akis = struct {
    w: *std.Io.Writer,
    a: std.mem.Allocator,
    id: []const u8,
    model: []const u8,
    created: i64,
    chat: bool,
    metin: ArrayList(u8),
    arac: bool = false, // tools verildi: <tool_call> sonrası akıtılmaz, sonda tool_calls olarak gider
    gonderilen: usize = 0,

    fn parca(ctx: *anyopaque, t: []const u8) anyerror!void {
        const k: *Akis = @ptrCast(@alignCast(ctx));
        try k.metin.appendSlice(t);
        if (k.id.len == 0) return; // stream değil: yalnız biriktir
        if (!k.arac) return k.gonder(t, null);
        const m = k.metin.items;
        var lim = std.mem.indexOf(u8, m, "<tool_call>") orelse m.len;
        if (lim == m.len) { // "<tool_call>" başı olabilecek kuyruğu tut
            var j: usize = @min(m.len, "<tool_call>".len - 1);
            while (j > 0) : (j -= 1) if (std.mem.startsWith(u8, "<tool_call>", m[m.len - j ..])) break;
            lim = m.len - j;
        }
        if (lim > k.gonderilen) {
            try k.gonder(m[k.gonderilen..lim], null);
            k.gonderilen = lim;
        }
    }
    fn gonder(k: *Akis, t: []const u8, finish: ?[]const u8) !void {
        const b = if (k.chat)
            try std.json.Stringify.valueAlloc(k.a, .{ .id = k.id, .object = "chat.completion.chunk", .created = k.created, .model = k.model, .choices = .{.{ .index = 0, .delta = .{ .content = t }, .finish_reason = finish }} }, .{})
        else
            try std.json.Stringify.valueAlloc(k.a, .{ .id = k.id, .object = "text_completion", .created = k.created, .model = k.model, .choices = .{.{ .index = 0, .text = t, .finish_reason = finish }} }, .{});
        try k.w.print("data: {s}\n\n", .{b});
        try k.w.flush();
    }
};

fn handle(s: anytype, conn: std.Io.net.Stream, model_name: []const u8) !void {
    var arena_s = std.heap.ArenaAllocator.init(s.a);
    defer arena_s.deinit();
    const a = arena_s.allocator();
    var rb: [1 << 16]u8 = undefined;
    var wb: [1 << 14]u8 = undefined;
    var rd = conn.reader(s.io, &rb);
    var wr = conn.writer(s.io, &wb);
    const w = &wr.interface;
    const req = readReq(&rd.interface, a) catch return hata(w, "400 Bad Request", "istek okunamadı", a);
    const now = std.Io.Timestamp.now(s.io, .real).toSeconds();

    if (std.mem.eql(u8, req.method, "OPTIONS")) {
        try w.writeAll("HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n");
        return w.flush();
    }
    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/health"))
        return yanit(w, "200 OK", "application/json", "{\"status\":\"ok\"}");
    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/v1/models")) {
        const b = try std.json.Stringify.valueAlloc(a, .{ .object = "list", .data = .{.{ .id = model_name, .object = "model", .created = now, .owned_by = "echo-core" }} }, .{});
        return yanit(w, "200 OK", "application/json", b);
    }
    const chat = std.mem.eql(u8, req.path, "/v1/chat/completions");
    if (!std.mem.eql(u8, req.method, "POST") or !(chat or std.mem.eql(u8, req.path, "/v1/completions")))
        return hata(w, "404 Not Found", "bilinmeyen yol", a);

    const parsed = std.json.parseFromSlice(Value, a, req.body, .{}) catch return hata(w, "400 Bad Request", "geçersiz JSON", a);
    if (parsed.value != .object) return hata(w, "400 Bad Request", "JSON nesne olmalı", a);
    const o = parsed.value.object;

    // örnekleme: istekte yoksa sunucu açılış ayarları
    var sp = s.smp.p;
    if (num(o.get("temperature"))) |x| sp.temp = @floatCast(x);
    if (num(o.get("top_p"))) |x| sp.top_p = @floatCast(x);
    if (num(o.get("top_k"))) |x| sp.top_k = @intFromFloat(x);
    if (num(o.get("min_p"))) |x| sp.min_p = @floatCast(x);
    if (num(o.get("presence_penalty"))) |x| sp.presence_penalty = @floatCast(x);
    if (num(o.get("repeat_penalty"))) |x| sp.repeat_penalty = @floatCast(x);
    if (num(o.get("seed"))) |x| sp.seed = @intFromFloat(x) else sp.seed +%= 1;
    if (s.ctx.lm != null) sp.temp = 0;
    s.smp = sampler.Sampler.init(sp);
    const max_t = num(o.get("max_tokens")) orelse num(o.get("max_completion_tokens")) orelse num(o.get("n_predict"));
    s.limit = if (max_t) |x| @intFromFloat(x) else -1;
    var stops = ArrayList([]const u8).init(a);
    if (o.get("stop")) |v| switch (v) {
        .string => |t| try stops.append(t),
        .array => |arr| for (arr.items) |x| if (str(x)) |t| try stops.append(t),
        else => {},
    };
    s.stops = stops.items;
    const stream = if (o.get("stream")) |v| v == .bool and v.bool else false;

    // prompt → token
    var ids = ArrayList(u32).init(a);
    var tools: ?Value = null;
    if (chat) {
        const msgs = o.get("messages") orelse return hata(w, "400 Bad Request", "messages yok", a);
        if (msgs != .array) return hata(w, "400 Bad Request", "messages dizi olmalı", a);
        var think = false;
        if (o.get("chat_template_kwargs")) |k| if (k == .object) if (k.object.get("enable_thinking")) |e| {
            think = e == .bool and e.bool;
        };
        // şablon için sayılar metin olarak (HF tojson ile aynı yazım)
        const pv = (std.json.parseFromSlice(Value, a, req.body, .{ .parse_numbers = false }) catch return hata(w, "400 Bad Request", "geçersiz JSON", a)).value.object;
        tools = pv.get("tools");
        if (tools) |t| if (t != .array or t.array.items.len == 0) {
            tools = null;
        };
        const metin = try sablon.kur(a, pv.get("messages").?.array.items, tools, think);
        if (std.c.getenv("ECHO_SABLON_YAZ")) |yol| std.Io.Dir.cwd().writeFile(s.io, .{ .sub_path = std.mem.span(yol), .data = metin }) catch {};
        try s.tokenize(metin, true, &ids);
    } else {
        const p = str(o.get("prompt")) orelse return hata(w, "400 Bad Request", "prompt yok", a);
        try s.tokenize(p, true, &ids);
    }
    if (ids.items.len + 1 > s.ctx.n_ctx) return hata(w, "400 Bad Request", "prompt context'ten uzun (-c ile büyütün)", a);

    // önek yeniden kullanımı: önceki isteğin son <|im_start|>'ına kadarki kısmı aynıysa oradan devam
    var bas: usize = 0;
    const k = &s.kayit.?;
    if (k.head > 0 and k.head < ids.items.len and std.mem.startsWith(u32, ids.items, s.kayit_ids.items)) {
        s.ctx.yukle(k);
        s.hist.shrinkRetainingCapacity(k.head);
        bas = k.head;
    } else {
        s.ctx.clear();
        s.hist.clearRetainingCapacity();
    }
    // bir sonraki istek için kayıt noktası: son <|im_start|> (asistan başlığı) öncesi
    s.kayit_noktasi = 0;
    if (chat) {
        const im_start = s.bpe.specialId("<|im_start|>") orelse 0;
        var j = ids.items.len;
        while (j > bas) : (j -= 1) if (ids.items[j - 1] == im_start) {
            s.kayit_noktasi = j - 1 - bas;
            break;
        };
    }
    const id = try std.fmt.allocPrint(a, "{s}-echo-{d}", .{ if (chat) "chatcmpl" else "cmpl", now });
    var ak = Akis{ .w = w, .a = a, .id = if (stream) id else "", .model = model_name, .created = now, .chat = chat, .metin = ArrayList(u8).init(a), .arac = tools != null };
    s.sink = .{ .ctx = &ak, .f = Akis.parca };
    defer s.sink = null;
    if (stream) try w.writeAll("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    if (stream and chat) try ak.gonder("", null); // ilk chunk: rol
    var r = try s.runIds(ids.items[bas..]);
    r.n_prompt = ids.items.len;
    var finish: []const u8 = if (std.mem.eql(u8, r.neden, "-n sınırı") or std.mem.eql(u8, r.neden, "context dolu")) "length" else "stop";
    std.debug.print("[{s} {s}: prompt {d} tok ({d} hazır), üretim {d} tok, {d:.1} tok/s, {s}]\n", .{ req.method, req.path, r.n_prompt, bas, r.n_gen, @as(f64, @floatFromInt(r.n_gen)) / (@as(f64, @floatFromInt(@max(r.t_gen, 1))) / 1e9), r.neden });
    const ay = if (tools != null) try sablon.ayir(a, ak.metin.items, tools) else sablon.Ayrik{ .content = ak.metin.items, .calls = &.{} };
    const Tc = struct { index: usize, id: []const u8, type: []const u8 = "function", function: sablon.Cagri };
    const tcs = try a.alloc(Tc, ay.calls.len);
    for (ay.calls, 0..) |c, i| tcs[i] = .{ .index = i, .id = try std.fmt.allocPrint(a, "call_{d}_{d}", .{ now, i }), .function = c };
    if (tcs.len > 0) finish = "tool_calls";
    if (stream) {
        ak.id = id;
        if (tcs.len > 0) {
            const b = try std.json.Stringify.valueAlloc(a, .{ .id = id, .object = "chat.completion.chunk", .created = now, .model = model_name, .choices = .{.{ .index = 0, .delta = .{ .tool_calls = tcs }, .finish_reason = null }} }, .{});
            try w.print("data: {s}\n\n", .{b});
        } else if (ak.arac and ak.gonderilen < ak.metin.items.len) try ak.gonder(ak.metin.items[ak.gonderilen..], null);
        try ak.gonder("", finish);
        try w.writeAll("data: [DONE]\n\n");
        return w.flush();
    }
    const usage = .{ .prompt_tokens = r.n_prompt, .completion_tokens = r.n_gen, .total_tokens = r.n_prompt + r.n_gen };
    const b = if (chat)
        try std.json.Stringify.valueAlloc(a, .{ .id = id, .object = "chat.completion", .created = now, .model = model_name, .choices = .{.{ .index = 0, .message = .{ .role = "assistant", .content = ay.content, .tool_calls = if (tcs.len > 0) tcs else null }, .finish_reason = finish }}, .usage = usage }, .{ .emit_null_optional_fields = false })
    else
        try std.json.Stringify.valueAlloc(a, .{ .id = id, .object = "text_completion", .created = now, .model = model_name, .choices = .{.{ .index = 0, .text = ak.metin.items, .finish_reason = finish }}, .usage = usage }, .{});
    try yanit(w, "200 OK", "application/json", b);
}
