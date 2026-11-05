//! Elle yazılmış ince Vulkan 1.1 compute bağlaması: libvulkan dlopen ile, tek kuyruk, tek arena.
//! shortcut: yalnız bu motorun kullandığı alanlar var; başka Vulkan özelliği gerekirse buraya eklenir.
const std = @import("std");

extern "c" fn dlopen(path: [*:0]const u8, flags: c_int) ?*anyopaque;
extern "c" fn dlsym(handle: ?*anyopaque, symbol: [*:0]const u8) ?*anyopaque;

pub const Disp = ?*anyopaque;
pub const H = u64;
const Cstr = ?[*:0]const u8;
const Pn = ?*const anyopaque;

pub const DESC_STORAGE: u32 = 7;
pub const USAGE_SRC: u32 = 1;
pub const USAGE_DST: u32 = 2;
pub const USAGE_STORAGE: u32 = 0x20;
pub const USAGE_INDIRECT: u32 = 0x100;
pub const STAGE_COMPUTE: u32 = 0x20;
pub const BIND_COMPUTE: u32 = 1;
pub const PIPE_TRANSFER: u32 = 0x1000;
pub const PIPE_COMPUTE: u32 = 0x800;
pub const PIPE_HOST: u32 = 0x4000;
pub const PIPE_INDIRECT: u32 = 0x2;
pub const ACC_INDIRECT: u32 = 0x1;
pub const ACC_UNIFORM_READ: u32 = 0x8;
pub const ACC_SHADER_READ: u32 = 0x20;
pub const ACC_SHADER_WRITE: u32 = 0x40;
pub const ACC_XFER_READ: u32 = 0x800;
pub const ACC_XFER_WRITE: u32 = 0x1000;
pub const ACC_HOST_READ: u32 = 0x2000;
pub const ACC_HOST_WRITE: u32 = 0x4000;
pub const MEM_LOCAL: u32 = 1;
pub const MEM_VISIBLE: u32 = 2;
pub const MEM_COHERENT: u32 = 4;
pub const MEM_CACHED: u32 = 8;

const ApplicationInfo = extern struct { sType: i32 = 0, pNext: Pn = null, name: Cstr = null, ver: u32 = 0, engine: Cstr = null, engine_ver: u32 = 0, api: u32 };
const InstanceCreateInfo = extern struct { sType: i32 = 1, pNext: Pn = null, flags: u32 = 0, app: *const ApplicationInfo, nl: u32 = 0, l: Pn = null, ne: u32 = 0, e: Pn = null };
const DeviceQueueCreateInfo = extern struct { sType: i32 = 2, pNext: Pn = null, flags: u32 = 0, family: u32, count: u32 = 1, prio: *const f32 };
const DeviceCreateInfo = extern struct { sType: i32 = 3, pNext: Pn = null, flags: u32 = 0, nq: u32 = 1, q: *const DeviceQueueCreateInfo, nl: u32 = 0, l: Pn = null, ne: u32 = 0, e: Pn = null, feat: Pn = null };
const QueueFamilyProperties = extern struct { flags: u32, count: u32, ts: u32, g: [3]u32 };
const MemoryType = extern struct { flags: u32, heap: u32 };
const MemoryHeap = extern struct { size: u64, flags: u32 };
const MemProps = extern struct { nt: u32, types: [32]MemoryType, nh: u32, heaps: [16]MemoryHeap };
const BufferCreateInfo = extern struct { sType: i32 = 12, pNext: Pn = null, flags: u32 = 0, size: u64, usage: u32, sharing: u32 = 0, nf: u32 = 0, f: Pn = null };
const MemoryRequirements = extern struct { size: u64, alignment: u64, bits: u32 };
const MemoryAllocateInfo = extern struct { sType: i32 = 5, pNext: Pn = null, size: u64, type: u32 };
const ShaderModuleCreateInfo = extern struct { sType: i32 = 16, pNext: Pn = null, flags: u32 = 0, size: usize, code: [*]const u32 };
const DescriptorSetLayoutBinding = extern struct { binding: u32, type: u32, count: u32, stages: u32, s: Pn = null };
const DescriptorSetLayoutCreateInfo = extern struct { sType: i32 = 32, pNext: Pn = null, flags: u32 = 0, n: u32, b: [*]const DescriptorSetLayoutBinding };
const PushConstantRange = extern struct { stages: u32, offset: u32, size: u32 };
const PipelineLayoutCreateInfo = extern struct { sType: i32 = 30, pNext: Pn = null, flags: u32 = 0, ns: u32, s: *const H, np: u32, p: *const PushConstantRange };
const PipelineShaderStageCreateInfo = extern struct { sType: i32 = 18, pNext: Pn = null, flags: u32 = 0, stage: u32, module: H, name: [*:0]const u8, spec: Pn = null };
const ComputePipelineCreateInfo = extern struct { sType: i32 = 29, pNext: Pn = null, flags: u32 = 0, stage: PipelineShaderStageCreateInfo, layout: H, base: H = 0, base_i: i32 = -1 };
const DescriptorPoolSize = extern struct { type: u32, count: u32 };
const DescriptorPoolCreateInfo = extern struct { sType: i32 = 33, pNext: Pn = null, flags: u32 = 0, max: u32, n: u32, s: *const DescriptorPoolSize };
const DescriptorSetAllocateInfo = extern struct { sType: i32 = 34, pNext: Pn = null, pool: H, n: u32 = 1, l: *const H };
const DescriptorBufferInfo = extern struct { buffer: H, offset: u64, range: u64 };
const WriteDescriptorSet = extern struct { sType: i32 = 35, pNext: Pn = null, set: H, binding: u32, arr: u32 = 0, n: u32 = 1, type: u32, img: Pn = null, buf: *const DescriptorBufferInfo, view: Pn = null };
const CommandPoolCreateInfo = extern struct { sType: i32 = 39, pNext: Pn = null, flags: u32 = 2, family: u32 };
const CommandBufferAllocateInfo = extern struct { sType: i32 = 40, pNext: Pn = null, pool: H, level: u32 = 0, n: u32 = 1 };
const CommandBufferBeginInfo = extern struct { sType: i32 = 42, pNext: Pn = null, flags: u32, inh: Pn = null };
pub const MemoryBarrier = extern struct { sType: i32 = 46, pNext: Pn = null, src: u32, dst: u32 };
pub const BufferCopy = extern struct { src: u64, dst: u64, size: u64 };
const SubmitInfo = extern struct { sType: i32 = 4, pNext: Pn = null, nw: u32 = 0, w: Pn = null, wm: Pn = null, nc: u32 = 1, c: [*]const Disp, ns: u32 = 0, s: Pn = null };
const QueryPoolCreateInfo = extern struct { sType: i32 = 11, pNext: Pn = null, flags: u32 = 0, type: u32 = 2, count: u32, stats: u32 = 0 };
const FenceCreateInfo = extern struct { sType: i32 = 8, pNext: Pn = null, flags: u32 = 0 };

const R = i32;
const Fns = struct {
    vkEnumeratePhysicalDevices: *const fn (Disp, *u32, ?[*]Disp) callconv(.c) R,
    vkGetPhysicalDeviceQueueFamilyProperties: *const fn (Disp, *u32, ?[*]QueueFamilyProperties) callconv(.c) void,
    vkGetPhysicalDeviceMemoryProperties: *const fn (Disp, *MemProps) callconv(.c) void,
    vkCreateDevice: *const fn (Disp, *const DeviceCreateInfo, Pn, *Disp) callconv(.c) R,
    vkGetDeviceQueue: *const fn (Disp, u32, u32, *Disp) callconv(.c) void,
    vkCreateBuffer: *const fn (Disp, *const BufferCreateInfo, Pn, *H) callconv(.c) R,
    vkGetBufferMemoryRequirements: *const fn (Disp, H, *MemoryRequirements) callconv(.c) void,
    vkAllocateMemory: *const fn (Disp, *const MemoryAllocateInfo, Pn, *H) callconv(.c) R,
    vkBindBufferMemory: *const fn (Disp, H, H, u64) callconv(.c) R,
    vkMapMemory: *const fn (Disp, H, u64, u64, u32, *?*anyopaque) callconv(.c) R,
    vkCreateShaderModule: *const fn (Disp, *const ShaderModuleCreateInfo, Pn, *H) callconv(.c) R,
    vkCreateDescriptorSetLayout: *const fn (Disp, *const DescriptorSetLayoutCreateInfo, Pn, *H) callconv(.c) R,
    vkCreatePipelineLayout: *const fn (Disp, *const PipelineLayoutCreateInfo, Pn, *H) callconv(.c) R,
    vkCreateComputePipelines: *const fn (Disp, H, u32, *const ComputePipelineCreateInfo, Pn, *H) callconv(.c) R,
    vkCreateDescriptorPool: *const fn (Disp, *const DescriptorPoolCreateInfo, Pn, *H) callconv(.c) R,
    vkAllocateDescriptorSets: *const fn (Disp, *const DescriptorSetAllocateInfo, *H) callconv(.c) R,
    vkUpdateDescriptorSets: *const fn (Disp, u32, [*]const WriteDescriptorSet, u32, Pn) callconv(.c) void,
    vkCreateCommandPool: *const fn (Disp, *const CommandPoolCreateInfo, Pn, *H) callconv(.c) R,
    vkAllocateCommandBuffers: *const fn (Disp, *const CommandBufferAllocateInfo, *Disp) callconv(.c) R,
    vkBeginCommandBuffer: *const fn (Disp, *const CommandBufferBeginInfo) callconv(.c) R,
    vkEndCommandBuffer: *const fn (Disp) callconv(.c) R,
    vkCmdBindPipeline: *const fn (Disp, u32, H) callconv(.c) void,
    vkCmdBindDescriptorSets: *const fn (Disp, u32, H, u32, u32, *const H, u32, Pn) callconv(.c) void,
    vkCmdPushConstants: *const fn (Disp, H, u32, u32, u32, *const anyopaque) callconv(.c) void,
    vkCmdDispatch: *const fn (Disp, u32, u32, u32) callconv(.c) void,
    vkCmdPipelineBarrier: *const fn (Disp, u32, u32, u32, u32, *const MemoryBarrier, u32, Pn, u32, Pn) callconv(.c) void,
    vkCmdCopyBuffer: *const fn (Disp, H, H, u32, *const BufferCopy) callconv(.c) void,
    vkCreateQueryPool: *const fn (Disp, *const QueryPoolCreateInfo, Pn, *H) callconv(.c) R,
    vkCmdResetQueryPool: *const fn (Disp, H, u32, u32) callconv(.c) void,
    vkCmdWriteTimestamp: *const fn (Disp, u32, H, u32) callconv(.c) void,
    vkGetQueryPoolResults: *const fn (Disp, H, u32, u32, usize, *anyopaque, u64, u32) callconv(.c) R,
    vkCmdDispatchIndirect: *const fn (Disp, H, u64) callconv(.c) void,
    vkCmdFillBuffer: *const fn (Disp, H, u64, u64, u32) callconv(.c) void,
    vkQueueSubmit: *const fn (Disp, u32, *const SubmitInfo, H) callconv(.c) R,
    vkCreateFence: *const fn (Disp, *const FenceCreateInfo, Pn, *H) callconv(.c) R,
    vkWaitForFences: *const fn (Disp, u32, *const H, u32, u64) callconv(.c) R,
    vkResetFences: *const fn (Disp, u32, *const H) callconv(.c) R,
    vkDeviceWaitIdle: *const fn (Disp) callconv(.c) R,
    vkDestroyDevice: *const fn (Disp, Pn) callconv(.c) void,
    vkDestroyInstance: *const fn (Disp, Pn) callconv(.c) void,
};

fn check(r: R) !void {
    if (r != 0) {
        std.debug.print("vulkan hata kodu: {d}\n", .{r});
        return error.VulkanFailed;
    }
}

pub const Buffer = struct { handle: H, map: ?[*]u8, size: u64 };

pub const Ctx = struct {
    inst: Disp,
    phys: Disp,
    dev: Disp,
    queue: Disp,
    family: u32,
    mem: MemProps,
    f: Fns,
    set_layout: H,
    layout: H,
    pool: H,
    set: H,
    cmd_pool: H,
    fence: H,

    pub const N_BIND = 2;
    pub const PC_SIZE = 64;

    pub fn init() !Ctx {
        const lib = dlopen("libvulkan.so.1", 2) orelse return error.VulkanNotFound;
        const gipa: *const fn (Disp, [*:0]const u8) callconv(.c) ?*anyopaque = @ptrCast(dlsym(lib, "vkGetInstanceProcAddr") orelse return error.VulkanNotFound);
        const create: *const fn (*const InstanceCreateInfo, Pn, *Disp) callconv(.c) R = @ptrCast(gipa(null, "vkCreateInstance") orelse return error.VulkanNotFound);
        const app = ApplicationInfo{ .api = (1 << 22) | (1 << 12) };
        var inst: Disp = null;
        try check(create(&.{ .app = &app }, null, &inst));
        var f: Fns = undefined;
        inline for (@typeInfo(Fns).@"struct".fields) |fd| {
            @field(f, fd.name) = @ptrCast(gipa(inst, fd.name ++ "") orelse return error.VulkanSymbolMissing);
        }

        var n: u32 = 1;
        var phys: Disp = null;
        const r0 = f.vkEnumeratePhysicalDevices(inst, &n, @ptrCast(&phys));
        if (r0 != 0 and r0 != 5) try check(r0); // VK_INCOMPLETE: ilk cihaz yeterli
        if (n == 0) return error.NoGpu;

        var nq: u32 = 16;
        var qf: [16]QueueFamilyProperties = undefined;
        f.vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, &qf);
        var family: u32 = 0;
        for (qf[0..nq], 0..) |q, i| {
            if (q.flags & 2 != 0) {
                family = @intCast(i);
                break;
            }
        }
        var mem: MemProps = undefined;
        f.vkGetPhysicalDeviceMemoryProperties(phys, &mem);

        const prio: f32 = 1.0;
        const qci = DeviceQueueCreateInfo{ .family = family, .prio = &prio };
        var dev: Disp = null;
        try check(f.vkCreateDevice(phys, &.{ .q = &qci }, null, &dev));
        var queue: Disp = null;
        f.vkGetDeviceQueue(dev, family, 0, &queue);

        const binds = [N_BIND]DescriptorSetLayoutBinding{
            .{ .binding = 0, .type = DESC_STORAGE, .count = 1, .stages = STAGE_COMPUTE },
            .{ .binding = 1, .type = DESC_STORAGE, .count = 1, .stages = STAGE_COMPUTE },
        };
        var set_layout: H = 0;
        try check(f.vkCreateDescriptorSetLayout(dev, &.{ .n = N_BIND, .b = &binds }, null, &set_layout));
        const pcr = PushConstantRange{ .stages = STAGE_COMPUTE, .offset = 0, .size = PC_SIZE };
        var layout: H = 0;
        try check(f.vkCreatePipelineLayout(dev, &.{ .ns = 1, .s = &set_layout, .np = 1, .p = &pcr }, null, &layout));
        const ps = DescriptorPoolSize{ .type = DESC_STORAGE, .count = N_BIND };
        var pool: H = 0;
        try check(f.vkCreateDescriptorPool(dev, &.{ .max = 1, .n = 1, .s = &ps }, null, &pool));
        var set: H = 0;
        try check(f.vkAllocateDescriptorSets(dev, &.{ .pool = pool, .l = &set_layout }, &set));
        var cmd_pool: H = 0;
        try check(f.vkCreateCommandPool(dev, &.{ .family = family }, null, &cmd_pool));
        var fence: H = 0;
        try check(f.vkCreateFence(dev, &.{}, null, &fence));
        return .{ .inst = inst, .phys = phys, .dev = dev, .queue = queue, .family = family, .mem = mem, .f = f, .set_layout = set_layout, .layout = layout, .pool = pool, .set = set, .cmd_pool = cmd_pool, .fence = fence };
    }

    /// shortcut: kaynaklar tek tek yok edilmez, süreç sonunda cihazla birlikte gider; uzun süreç içinde yeniden kurulursa ayrıca yıkım gerekir.
    pub fn deinit(self: *Ctx) void {
        _ = self.f.vkDeviceWaitIdle(self.dev);
        self.f.vkDestroyDevice(self.dev, null);
        self.f.vkDestroyInstance(self.inst, null);
    }

    fn memType(self: *const Ctx, bits: u32, want: u32) !u32 {
        for (0..self.mem.nt) |i| {
            if (bits & (@as(u32, 1) << @intCast(i)) != 0 and self.mem.types[i].flags & want == want) return @intCast(i);
        }
        return error.NoMemoryType;
    }

    pub fn buffer(self: *const Ctx, size: u64, usage: u32, want: u32) !Buffer {
        var b: H = 0;
        try check(self.f.vkCreateBuffer(self.dev, &.{ .size = size, .usage = usage }, null, &b));
        var req: MemoryRequirements = undefined;
        self.f.vkGetBufferMemoryRequirements(self.dev, b, &req);
        var mem: H = 0;
        try check(self.f.vkAllocateMemory(self.dev, &.{ .size = req.size, .type = try self.memType(req.bits, want) }, null, &mem));
        try check(self.f.vkBindBufferMemory(self.dev, b, mem, 0));
        var map: ?*anyopaque = null;
        if (want & MEM_VISIBLE != 0) try check(self.f.vkMapMemory(self.dev, mem, 0, ~@as(u64, 0), 0, &map));
        return .{ .handle = b, .map = @ptrCast(map), .size = size };
    }

    pub fn bind(self: *const Ctx, binding: u32, b: Buffer) void {
        const info = DescriptorBufferInfo{ .buffer = b.handle, .offset = 0, .range = b.size };
        const w = [1]WriteDescriptorSet{.{ .set = self.set, .binding = binding, .type = DESC_STORAGE, .buf = &info }};
        self.f.vkUpdateDescriptorSets(self.dev, 1, &w, 0, null);
    }

    pub fn pipeline(self: *const Ctx, spv: []const u8) !H {
        var m: H = 0;
        try check(self.f.vkCreateShaderModule(self.dev, &.{ .size = spv.len, .code = @ptrCast(@alignCast(spv.ptr)) }, null, &m));
        const ci = ComputePipelineCreateInfo{
            .stage = .{ .stage = STAGE_COMPUTE, .module = m, .name = "main" },
            .layout = self.layout,
        };
        var p: H = 0;
        try check(self.f.vkCreateComputePipelines(self.dev, 0, 1, &ci, null, &p));
        return p;
    }

    pub fn queryPool(self: *const Ctx, n: u32) !H {
        var q: H = 0;
        try check(self.f.vkCreateQueryPool(self.dev, &.{ .count = n }, null, &q));
        return q;
    }

    pub fn cmdBuffer(self: *const Ctx) !Disp {
        var c: Disp = null;
        try check(self.f.vkAllocateCommandBuffers(self.dev, &.{ .pool = self.cmd_pool }, &c));
        return c;
    }
    /// flags 0: tekrar gönderilebilir komut tamponu
    pub fn begin(self: *const Ctx, c: Disp) !void {
        try self.beginFlags(c, 0);
    }
    /// 4: SIMULTANEOUS_USE, aynı tampon tek gönderimde tekrar edilebilir
    pub fn beginFlags(self: *const Ctx, c: Disp, flags: u32) !void {
        try check(self.f.vkBeginCommandBuffer(c, &.{ .flags = flags }));
        self.f.vkCmdBindDescriptorSets(c, BIND_COMPUTE, self.layout, 0, 1, &self.set, 0, null);
    }
    pub fn end(self: *const Ctx, c: Disp) !void {
        try check(self.f.vkEndCommandBuffer(c));
    }
    pub fn submitWait(self: *const Ctx, c: Disp) !void {
        const cb = [1]Disp{c};
        try self.submitListWait(&cb);
    }
    pub fn submitListWait(self: *const Ctx, list: []const Disp) !void {
        try check(self.f.vkQueueSubmit(self.queue, 1, &.{ .nc = @intCast(list.len), .c = list.ptr }, self.fence));
        try check(self.f.vkWaitForFences(self.dev, 1, &self.fence, 1, ~@as(u64, 0)));
        try check(self.f.vkResetFences(self.dev, 1, &self.fence));
    }
    pub fn barrier(self: *const Ctx, c: Disp, src_stage: u32, dst_stage: u32, src: u32, dst: u32) void {
        const b = MemoryBarrier{ .src = src, .dst = dst };
        self.f.vkCmdPipelineBarrier(c, src_stage, dst_stage, 0, 1, &b, 0, null, 0, null);
    }
};
