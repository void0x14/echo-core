//! Qwen3.5 resmi chat şablonunun (GGUF tokenizer.chat_template) elle yazılmış karşılığı:
//! system birleştirme, tools, tool rolü, tool_calls geçmişi, son sorgudan sonraki düşünce bloğu.
//! Çıktı ayrıştırma: <tool_call><function=ad><parameter=p>değer</parameter></function></tool_call> → OpenAI tool_calls.
const std = @import("std");
const Value = std.json.Value;
const Alloc = std.mem.Allocator;
const Buf = std.ArrayList(u8);

const bosluk = " \t\n\r\x0b\x0c";

fn get(v: Value, k: []const u8) ?Value {
    if (v != .object) return null;
    const x = v.object.get(k) orelse return null;
    return if (x == .null) null else x;
}
fn str(v: ?Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}
fn rol(m: Value) []const u8 {
    return str(get(m, "role")) orelse "";
}
fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn sistemMi(m: Value) bool {
    return eq(rol(m), "system") or eq(rol(m), "developer");
}

/// render_content: metin ya da {text} parçaları (görsel/video yok)
fn icerik(a: Alloc, c: ?Value) ![]const u8 {
    const x = c orelse return "";
    switch (x) {
        .string => |t| return t,
        .array => |arr| {
            var b: Buf = .empty;
            for (arr.items) |p| if (str(get(p, "text"))) |t| try b.appendSlice(a, t);
            return b.items;
        },
        else => return "",
    }
}
fn kirp(t: []const u8) []const u8 {
    return std.mem.trim(u8, t, bosluk);
}

/// HF `tojson`: json.dumps(ensure_ascii=False) — ", " ve ": " ayraçları, anahtar sırası korunur.
/// Sayılar parse_numbers=false ile okunmuş olmalı (number_string olduğu gibi yazılır).
pub fn tojson(a: Alloc, b: *Buf, v: Value) !void {
    switch (v) {
        .null => try b.appendSlice(a, "null"),
        .bool => |x| try b.appendSlice(a, if (x) "true" else "false"),
        .integer => |x| try b.print(a, "{d}", .{x}),
        .float => |x| try b.print(a, "{d}", .{x}),
        .number_string => |x| try b.appendSlice(a, x),
        .string => |x| try jsonStr(a, b, x),
        .array => |arr| {
            try b.append(a, '[');
            for (arr.items, 0..) |x, i| {
                if (i > 0) try b.appendSlice(a, ", ");
                try tojson(a, b, x);
            }
            try b.append(a, ']');
        },
        .object => |o| {
            try b.append(a, '{');
            var it = o.iterator();
            var i: usize = 0;
            while (it.next()) |e| : (i += 1) {
                if (i > 0) try b.appendSlice(a, ", ");
                try jsonStr(a, b, e.key_ptr.*);
                try b.appendSlice(a, ": ");
                try tojson(a, b, e.value_ptr.*);
            }
            try b.append(a, '}');
        },
    }
}
fn jsonStr(a: Alloc, b: *Buf, s: []const u8) !void {
    try b.append(a, '"');
    for (s) |c| switch (c) {
        '"' => try b.appendSlice(a, "\\\""),
        '\\' => try b.appendSlice(a, "\\\\"),
        '\n' => try b.appendSlice(a, "\\n"),
        '\r' => try b.appendSlice(a, "\\r"),
        '\t' => try b.appendSlice(a, "\\t"),
        0x08 => try b.appendSlice(a, "\\b"),
        0x0c => try b.appendSlice(a, "\\f"),
        0...0x07, 0x0b, 0x0e...0x1f => try b.print(a, "\\u{x:0>4}", .{c}),
        else => try b.append(a, c),
    };
    try b.append(a, '"');
}

/// Python `str(x)`; shortcut: float Python repr'i değil JSON yazımı (1e5 gibi nadir biçimlerde fark), gerekirse repr yaz.
fn pyStr(a: Alloc, b: *Buf, v: Value) !void {
    switch (v) {
        .string => |x| try b.appendSlice(a, x),
        .null => try b.appendSlice(a, "None"),
        .bool => |x| try b.appendSlice(a, if (x) "True" else "False"),
        .object, .array => try tojson(a, b, v),
        else => try tojson(a, b, v),
    }
}

const arac_metni = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Resmi şablonla aynı metni üretir (add_generation_prompt=true).
/// msgs/tools sayıları parse_numbers=false ile okunmuş olmalı.
pub fn kur(a: Alloc, msgs: []const Value, tools: ?Value, think: bool) ![]u8 {
    var b: Buf = .empty;
    var num_sys: usize = 0;
    var merged: []const u8 = "";
    if (msgs.len > 0 and sistemMi(msgs[0])) {
        const first = kirp(try icerik(a, get(msgs[0], "content")));
        if (msgs.len > 1 and sistemMi(msgs[1])) {
            merged = try std.mem.concat(a, u8, &.{ first, "\n", kirp(try icerik(a, get(msgs[1], "content"))) });
            num_sys = 2;
        } else {
            merged = first;
            num_sys = 1;
        }
    }
    const araclar: []const Value = if (tools) |t| (if (t == .array) t.array.items else &.{}) else &.{};
    if (araclar.len > 0) {
        try b.appendSlice(a, "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for (araclar) |t| {
            try b.append(a, '\n');
            try tojson(a, &b, t);
        }
        try b.appendSlice(a, "\n</tools>");
        try b.appendSlice(a, arac_metni);
        if (merged.len > 0) {
            try b.appendSlice(a, "\n\n");
            try b.appendSlice(a, merged);
        }
        try b.appendSlice(a, "<|im_end|>\n");
    } else if (merged.len > 0) {
        try b.appendSlice(a, "<|im_start|>system\n");
        try b.appendSlice(a, merged);
        try b.appendSlice(a, "<|im_end|>\n");
    }

    // son gerçek kullanıcı sorgusu (tool_response sarmalı olmayan)
    var last_q: usize = msgs.len -| 1;
    var i = msgs.len;
    while (i > 0) {
        i -= 1;
        if (!eq(rol(msgs[i]), "user")) continue;
        const c = kirp(try icerik(a, get(msgs[i], "content")));
        if (!(std.mem.startsWith(u8, c, "<tool_response>") and std.mem.endsWith(u8, c, "</tool_response>"))) {
            last_q = i;
            break;
        }
    }

    for (msgs, 0..) |m, idx| {
        if (idx < num_sys or sistemMi(m)) continue;
        const r = rol(m);
        var content = kirp(try icerik(a, get(m, "content")));
        if (eq(r, "user")) {
            try b.appendSlice(a, "<|im_start|>user\n");
            try b.appendSlice(a, content);
            try b.appendSlice(a, "<|im_end|>\n");
        } else if (eq(r, "assistant")) {
            var reasoning: []const u8 = "";
            if (str(get(m, "reasoning_content"))) |rc| {
                reasoning = rc;
            } else if (std.mem.indexOf(u8, content, "</think>")) |e0| {
                const once = std.mem.trimEnd(u8, content[0..e0], "\n");
                reasoning = if (std.mem.lastIndexOf(u8, once, "<think>")) |s0| once[s0 + 7 ..] else once;
                reasoning = std.mem.trimStart(u8, reasoning, "\n");
                const son = std.mem.lastIndexOf(u8, content, "</think>").?;
                content = std.mem.trimStart(u8, content[son + 8 ..], "\n");
            }
            reasoning = kirp(reasoning);
            if (idx > last_q) {
                try b.appendSlice(a, "<|im_start|>assistant\n<think>\n");
                try b.appendSlice(a, reasoning);
                try b.appendSlice(a, "\n</think>\n\n");
            } else try b.appendSlice(a, "<|im_start|>assistant\n");
            try b.appendSlice(a, content);
            if (get(m, "tool_calls")) |tcs| if (tcs == .array) for (tcs.array.items, 0..) |tc0, k| {
                const tc = get(tc0, "function") orelse tc0;
                const ad = str(get(tc, "name")) orelse "";
                if (k == 0) {
                    try b.appendSlice(a, if (kirp(content).len > 0) "\n\n<tool_call>\n<function=" else "<tool_call>\n<function=");
                } else try b.appendSlice(a, "\n<tool_call>\n<function=");
                try b.appendSlice(a, ad);
                try b.appendSlice(a, ">\n");
                // OpenAI arguments JSON metni gönderir; llama.cpp gibi nesneye çevrilir
                var args = get(tc, "arguments");
                if (args) |x| if (x == .string) {
                    args = std.json.parseFromSliceLeaky(Value, a, x.string, .{ .parse_numbers = false }) catch null;
                };
                if (args) |x| if (x == .object) {
                    var it = x.object.iterator();
                    while (it.next()) |e| {
                        try b.appendSlice(a, "<parameter=");
                        try b.appendSlice(a, e.key_ptr.*);
                        try b.appendSlice(a, ">\n");
                        try pyStr(a, &b, e.value_ptr.*);
                        try b.appendSlice(a, "\n</parameter>\n");
                    }
                };
                try b.appendSlice(a, "</function>\n</tool_call>");
            };
            try b.appendSlice(a, "<|im_end|>\n");
        } else if (eq(r, "tool")) {
            if (idx > 0 and !eq(rol(msgs[idx - 1]), "tool")) try b.appendSlice(a, "<|im_start|>user");
            try b.appendSlice(a, "\n<tool_response>\n");
            try b.appendSlice(a, content);
            try b.appendSlice(a, "\n</tool_response>");
            if (idx + 1 == msgs.len or !eq(rol(msgs[idx + 1]), "tool")) try b.appendSlice(a, "<|im_end|>\n");
        }
    }
    try b.appendSlice(a, "<|im_start|>assistant\n");
    try b.appendSlice(a, if (think) "<think>\n" else "<think>\n\n</think>\n\n");
    return b.items;
}

pub const Cagri = struct { name: []const u8, arguments: []const u8 };
pub const Ayrik = struct { content: []const u8, calls: []Cagri };

/// Parametre şemadaki türü string değilse JSON olarak okunur (llama.cpp qwen3-coder ayrıştırıcısı gibi).
fn turString(tools: ?Value, fn_ad: []const u8, p: []const u8) bool {
    const t = tools orelse return true;
    if (t != .array) return true;
    for (t.array.items) |x| {
        const f = get(x, "function") orelse x;
        if (!eq(str(get(f, "name")) orelse "", fn_ad)) continue;
        const props = get(get(f, "parameters") orelse return true, "properties") orelse return true;
        const ps = get(props, p) orelse return true;
        const ty = get(ps, "type") orelse return false;
        return ty == .string and eq(ty.string, "string");
    }
    return true;
}

/// Üretilen metni içerik + araç çağrılarına ayırır. Çağrı yoksa content = metnin tamamı.
pub fn ayir(a: Alloc, metin: []const u8, tools: ?Value) !Ayrik {
    const ilk = std.mem.indexOf(u8, metin, "<tool_call>") orelse return .{ .content = metin, .calls = &.{} };
    var calls: std.ArrayList(Cagri) = .empty;
    var rest = metin[ilk..];
    while (std.mem.indexOf(u8, rest, "<tool_call>")) |s0| {
        const govde_bas = s0 + "<tool_call>".len;
        const son = std.mem.indexOfPos(u8, rest, govde_bas, "</tool_call>") orelse rest.len;
        const govde = rest[govde_bas..son];
        rest = rest[@min(son + "</tool_call>".len, rest.len)..];
        const fb = std.mem.indexOf(u8, govde, "<function=") orelse continue;
        const fe = std.mem.indexOfScalarPos(u8, govde, fb, '>') orelse continue;
        const ad = kirp(govde[fb + 10 .. fe]);
        var args: Buf = .empty;
        try args.append(a, '{');
        var n: usize = 0;
        var g = govde[fe + 1 ..];
        while (std.mem.indexOf(u8, g, "<parameter=")) |pb| {
            const pe = std.mem.indexOfScalarPos(u8, g, pb, '>') orelse break;
            const pad = kirp(g[pb + 11 .. pe]);
            const vs = pe + 1;
            const ve = std.mem.indexOfPos(u8, g, vs, "</parameter>") orelse
                std.mem.indexOfPos(u8, g, vs, "<parameter=") orelse
                std.mem.indexOfPos(u8, g, vs, "</function>") orelse g.len;
            var deger = g[vs..ve];
            if (std.mem.startsWith(u8, deger, "\n")) deger = deger[1..];
            if (std.mem.endsWith(u8, deger, "\n")) deger = deger[0 .. deger.len - 1];
            if (n > 0) try args.appendSlice(a, ", ");
            try jsonStr(a, &args, pad);
            try args.appendSlice(a, ": ");
            const ham = !turString(tools, ad, pad) and blk: {
                _ = std.json.parseFromSliceLeaky(Value, a, deger, .{}) catch break :blk false;
                break :blk true;
            };
            if (ham) try args.appendSlice(a, kirp(deger)) else try jsonStr(a, &args, deger);
            n += 1;
            g = g[@min(ve + (if (std.mem.startsWith(u8, g[ve..], "</parameter>")) "</parameter>".len else 0), g.len)..];
        }
        try args.append(a, '}');
        try calls.append(a, .{ .name = ad, .arguments = args.items });
    }
    if (calls.items.len == 0) return .{ .content = metin, .calls = &.{} };
    return .{ .content = std.mem.trimEnd(u8, metin[0..ilk], bosluk), .calls = calls.items };
}

test "ayir: tek çağrı, sayı ve metin parametresi" {
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const a = ar.allocator();
    const tools = try std.json.parseFromSliceLeaky(Value, a,
        \\[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"x":{"type":"integer"},"s":{"type":"string"}}}}}]
    , .{});
    const r = try ayir(a, "tamam\n\n<tool_call>\n<function=f>\n<parameter=x>\n42\n</parameter>\n<parameter=s>\n12\n</parameter>\n</function>\n</tool_call>", tools);
    try std.testing.expectEqualStrings("tamam", r.content);
    try std.testing.expectEqual(@as(usize, 1), r.calls.len);
    try std.testing.expectEqualStrings("f", r.calls[0].name);
    try std.testing.expectEqualStrings("{\"x\": 42, \"s\": \"12\"}", r.calls[0].arguments);
    const yok = try ayir(a, "düz cevap", tools);
    try std.testing.expectEqual(@as(usize, 0), yok.calls.len);
}
