import { type Img, createImg } from "./image.ts";

let device: GPUDevice | null = null;
let gpuAvailable: boolean | null = null;

async function getDevice(): Promise<GPUDevice | null> {
  if (gpuAvailable === false) return null;
  if (device) return device;

  try {
    if (!navigator.gpu) {
      gpuAvailable = false;
      return null;
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      gpuAvailable = false;
      return null;
    }
    device = await adapter.requestDevice();
    void device.lost.then(() => {
      device = null;
    });
    gpuAvailable = true;
    return device;
  } catch {
    gpuAvailable = false;
    return null;
  }
}

// ── Buffer Pool ──────────────────────────────────────────────────────────────
// Key by usage|size to reuse GPU buffers across dispatches within a pipeline run

const bufferPool = new Map<string, GPUBuffer[]>();

function poolKey(size: number, usage: number): string {
  return `${usage}|${size}`;
}

function acquireBuffer(dev: GPUDevice, size: number, usage: number): GPUBuffer {
  const key = poolKey(size, usage);
  const pool = bufferPool.get(key);
  if (pool && pool.length > 0) return pool.pop()!;
  return dev.createBuffer({ size, usage });
}

function releaseBuffer(buf: GPUBuffer, size: number, usage: number): void {
  const key = poolKey(size, usage);
  let pool = bufferPool.get(key);
  if (!pool) {
    pool = [];
    bufferPool.set(key, pool);
  }
  pool.push(buf);
}

/** Call between pipeline runs to prevent unbounded growth */
export function flushBufferPool(): void {
  for (const pool of bufferPool.values()) {
    for (const buf of pool) buf.destroy();
  }
  bufferPool.clear();
}

const STORAGE_DST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const STORAGE_SRC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
const READ_DST = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
const UNIFORM_DST = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

function allocAndWrite(dev: GPUDevice, data: BufferSource, usage: number): GPUBuffer {
  const buf = acquireBuffer(dev, (data as ArrayBuffer).byteLength, usage);
  dev.queue.writeBuffer(buf, 0, data);
  return buf;
}

async function readBackFloat32(
  dev: GPUDevice,
  src: GPUBuffer,
  size: number,
): Promise<Float32Array> {
  // Don't pool MAP_READ buffers — they have mapping state that causes issues
  // with reuse, especially under parallel dispatches
  const readBuf = dev.createBuffer({ size, usage: READ_DST });
  const encoder = dev.createCommandEncoder();
  encoder.copyBufferToBuffer(src, 0, readBuf, 0, size);
  dev.queue.submit([encoder.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const mapped = new Float32Array(readBuf.getMappedRange());
  const result = new Float32Array(mapped.length);
  result.set(mapped);
  readBuf.unmap();
  readBuf.destroy();
  return result;
}

function submitCompute(
  dev: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroups: number,
): void {
  const encoder = dev.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  dev.queue.submit([encoder.finish()]);
}

// ── GPU Roll ─────────────────────────────────────────────────────────────────

const ROLL_SHADER = /* wgsl */ `
struct Params { w: u32, h: u32, c: u32, _p: u32, dx: i32, dy: i32, _p2: u32, _p3: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x; if (px >= params.w * params.h) { return; }
  let y = i32(px / params.w); let x = i32(px % params.w);
  let w = i32(params.w); let h = i32(params.h);
  let sy = ((y - params.dy) % h + h) % h;
  let sx = ((x - params.dx) % w + w) % w;
  let si = u32(sy * w + sx);
  if (params.c == 1u) {
    output[px] = input[si];
  } else {
    let oi = px * 3u; let ii = si * 3u;
    output[oi] = input[ii]; output[oi+1u] = input[ii+1u]; output[oi+2u] = input[ii+2u];
  }
}`;

let rollPipeline: GPUComputePipeline | null = null;

// ── GPU Convolution (sparse) ─────────────────────────────────────────────────

const CONVOLVE_SHADER = /* wgsl */ `
struct Params { w: u32, h: u32, c: u32, numTaps: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> taps: array<f32>;

fn rc(v: i32, mx: i32) -> i32 {
  var r = v; if (r < 0) { r = -r; }
  if (r >= mx) { let p = mx * 2; r = r % p; if (r >= mx) { r = p - r - 1; } }
  return r;
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x; if (px >= params.w * params.h) { return; }
  let y = i32(px / params.w); let x = i32(px % params.w);
  let w = i32(params.w); let h = i32(params.h); let n = params.numTaps;
  if (params.c == 1u) {
    var s: f32 = 0.0;
    for (var i = 0u; i < n; i++) {
      s += input[u32(rc(y + i32(taps[i*3u]), h) * w + rc(x + i32(taps[i*3u+1u]), w))] * taps[i*3u+2u];
    }
    output[px] = s;
  } else {
    var s0: f32 = 0.0; var s1: f32 = 0.0; var s2: f32 = 0.0;
    for (var i = 0u; i < n; i++) {
      let idx = u32(rc(y + i32(taps[i*3u]), h) * w + rc(x + i32(taps[i*3u+1u]), w)) * 3u;
      let wt = taps[i*3u+2u];
      s0 += input[idx] * wt; s1 += input[idx+1u] * wt; s2 += input[idx+2u] * wt;
    }
    let oi = px * 3u; output[oi] = s0; output[oi+1u] = s1; output[oi+2u] = s2;
  }
}`;

let convolvePipeline: GPUComputePipeline | null = null;

export async function gpuConvolve2d(
  img: Img,
  kernel: ArrayLike<number>,
  kw: number,
  kh: number,
): Promise<Img | null> {
  const dev = await getDevice();
  if (!dev) return null;
  const { w, h, c } = img;
  const khh = (kh - 1) >> 1,
    kwh = (kw - 1) >> 1;
  const tapsList: number[] = [];
  for (let ky = 0; ky < kh; ky++)
    for (let kx = 0; kx < kw; kx++) {
      const v = kernel[ky * kw + kx]!;
      if (v !== 0) tapsList.push(ky - khh, kx - kwh, v);
    }
  const numTaps = tapsList.length / 3;
  if (!convolvePipeline) {
    convolvePipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: CONVOLVE_SHADER }), entryPoint: "main" },
    });
  }
  const totalPx = w * h;
  const dataSize = totalPx * c * 4;
  const paramBuf = allocAndWrite(dev, new Uint32Array([w, h, c, numTaps]), UNIFORM_DST);
  const inputBuf = allocAndWrite(dev, img.data as unknown as BufferSource, STORAGE_DST);
  const outputBuf = acquireBuffer(dev, dataSize, STORAGE_SRC);
  const tapsBuf = allocAndWrite(
    dev,
    new Float32Array(tapsList) as unknown as BufferSource,
    STORAGE_DST,
  );
  const bg = dev.createBindGroup({
    layout: convolvePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: tapsBuf } },
    ],
  });
  submitCompute(dev, convolvePipeline, bg, Math.ceil(totalPx / 256));
  const result = await readBackFloat32(dev, outputBuf, dataSize);
  releaseBuffer(paramBuf, 16, UNIFORM_DST);
  releaseBuffer(inputBuf, dataSize, STORAGE_DST);
  releaseBuffer(outputBuf, dataSize, STORAGE_SRC);
  releaseBuffer(tapsBuf, tapsList.length * 4, STORAGE_DST);
  const out = createImg(w, h, c);
  out.data.set(result);
  return out;
}

// ── GPU Ripple (separable: rowDx + colDy, no full map upload) ────────────────

const RIPPLE_SHADER = /* wgsl */ `
struct Params { w: u32, h: u32, c: u32, _p: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> rowDx: array<f32>;
@group(0) @binding(4) var<storage, read> colDy: array<f32>;

fn rc(v: i32, mx: i32) -> i32 {
  var r = v; if (r < 0) { r = -r; }
  if (r >= mx) { let p = mx * 2; r = r % p; if (r >= mx) { r = p - r - 1; } }
  return r;
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x; if (px >= params.w * params.h) { return; }
  let y = i32(px / params.w); let x = i32(px % params.w);
  let w = i32(params.w); let h = i32(params.h);
  let fx = f32(x) + rowDx[y];
  let fy = f32(y) + colDy[x];
  let x0 = i32(floor(fx)); let y0 = i32(floor(fy));
  let wx = fx - f32(x0); let wy = fy - f32(y0);
  let rx0 = rc(x0, w); let rx1 = rc(x0+1, w);
  let ry0 = rc(y0, h); let ry1 = rc(y0+1, h);
  let w00 = (1.0-wx)*(1.0-wy); let w10 = wx*(1.0-wy);
  let w01 = (1.0-wx)*wy; let w11 = wx*wy;
  if (params.c == 1u) {
    output[px] = input[u32(ry0*w+rx0)]*w00 + input[u32(ry0*w+rx1)]*w10
               + input[u32(ry1*w+rx0)]*w01 + input[u32(ry1*w+rx1)]*w11;
  } else {
    let i00=u32(ry0*w+rx0)*3u; let i10=u32(ry0*w+rx1)*3u;
    let i01=u32(ry1*w+rx0)*3u; let i11=u32(ry1*w+rx1)*3u;
    let oi = px*3u;
    output[oi]   = input[i00]*w00+input[i10]*w10+input[i01]*w01+input[i11]*w11;
    output[oi+1u]= input[i00+1u]*w00+input[i10+1u]*w10+input[i01+1u]*w01+input[i11+1u]*w11;
    output[oi+2u]= input[i00+2u]*w00+input[i10+2u]*w10+input[i01+2u]*w01+input[i11+2u]*w11;
  }
}`;

let ripplePipeline: GPUComputePipeline | null = null;

export async function gpuRipple(
  img: Img,
  rowDx: Float32Array,
  colDy: Float32Array,
): Promise<Img | null> {
  const dev = await getDevice();
  if (!dev) return null;
  const { w, h, c } = img;
  const totalPx = w * h;
  const dataSize = totalPx * c * 4;
  if (!ripplePipeline) {
    ripplePipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: RIPPLE_SHADER }), entryPoint: "main" },
    });
  }
  const paramBuf = allocAndWrite(dev, new Uint32Array([w, h, c, 0]), UNIFORM_DST);
  const inputBuf = allocAndWrite(dev, img.data as unknown as BufferSource, STORAGE_DST);
  const outputBuf = acquireBuffer(dev, dataSize, STORAGE_SRC);
  const dxBuf = allocAndWrite(dev, rowDx as unknown as BufferSource, STORAGE_DST);
  const dyBuf = allocAndWrite(dev, colDy as unknown as BufferSource, STORAGE_DST);
  const bg = dev.createBindGroup({
    layout: ripplePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: dxBuf } },
      { binding: 4, resource: { buffer: dyBuf } },
    ],
  });
  submitCompute(dev, ripplePipeline, bg, Math.ceil(totalPx / 256));
  const result = await readBackFloat32(dev, outputBuf, dataSize);
  releaseBuffer(paramBuf, 16, UNIFORM_DST);
  releaseBuffer(inputBuf, dataSize, STORAGE_DST);
  releaseBuffer(outputBuf, dataSize, STORAGE_SRC);
  releaseBuffer(dxBuf, rowDx.byteLength, STORAGE_DST);
  releaseBuffer(dyBuf, colDy.byteLength, STORAGE_DST);
  const out = createImg(w, h, c);
  out.data.set(result);
  return out;
}

// ── GPU Remap (bilinear) ─────────────────────────────────────────────────────

const REMAP_SHADER = /* wgsl */ `
struct Params { w: u32, h: u32, c: u32, _p: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> mapX: array<f32>;
@group(0) @binding(4) var<storage, read> mapY: array<f32>;

fn rc(v: i32, mx: i32) -> i32 {
  var r = v; if (r < 0) { r = -r; }
  if (r >= mx) { let p = mx * 2; r = r % p; if (r >= mx) { r = p - r - 1; } }
  return r;
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x; if (px >= params.w * params.h) { return; }
  let w = i32(params.w); let h = i32(params.h);
  let fx = mapX[px]; let fy = mapY[px];
  let x0 = i32(floor(fx)); let y0 = i32(floor(fy));
  let wx = fx - f32(x0); let wy = fy - f32(y0);
  let rx0 = rc(x0, w); let rx1 = rc(x0+1, w);
  let ry0 = rc(y0, h); let ry1 = rc(y0+1, h);
  let w00 = (1.0-wx)*(1.0-wy); let w10 = wx*(1.0-wy);
  let w01 = (1.0-wx)*wy; let w11 = wx*wy;
  if (params.c == 1u) {
    output[px] = input[u32(ry0*w+rx0)]*w00 + input[u32(ry0*w+rx1)]*w10
               + input[u32(ry1*w+rx0)]*w01 + input[u32(ry1*w+rx1)]*w11;
  } else {
    let i00=u32(ry0*w+rx0)*3u; let i10=u32(ry0*w+rx1)*3u;
    let i01=u32(ry1*w+rx0)*3u; let i11=u32(ry1*w+rx1)*3u;
    let oi = px*3u;
    output[oi]   = input[i00]*w00+input[i10]*w10+input[i01]*w01+input[i11]*w11;
    output[oi+1u]= input[i00+1u]*w00+input[i10+1u]*w10+input[i01+1u]*w01+input[i11+1u]*w11;
    output[oi+2u]= input[i00+2u]*w00+input[i10+2u]*w10+input[i01+2u]*w01+input[i11+2u]*w11;
  }
}`;

let remapPipeline: GPUComputePipeline | null = null;

export async function gpuRemap(
  img: Img,
  mapXData: Float32Array,
  mapYData: Float32Array,
): Promise<Img | null> {
  const dev = await getDevice();
  if (!dev) return null;
  const { w, h, c } = img;
  const totalPx = w * h;
  const dataSize = totalPx * c * 4;
  const mapSize = totalPx * 4;
  if (!remapPipeline) {
    remapPipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: REMAP_SHADER }), entryPoint: "main" },
    });
  }
  const paramBuf = allocAndWrite(dev, new Uint32Array([w, h, c, 0]), UNIFORM_DST);
  const inputBuf = allocAndWrite(dev, img.data as unknown as BufferSource, STORAGE_DST);
  const outputBuf = acquireBuffer(dev, dataSize, STORAGE_SRC);
  const mapXBuf = allocAndWrite(dev, mapXData as unknown as BufferSource, STORAGE_DST);
  const mapYBuf = allocAndWrite(dev, mapYData as unknown as BufferSource, STORAGE_DST);
  const bg = dev.createBindGroup({
    layout: remapPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: mapXBuf } },
      { binding: 4, resource: { buffer: mapYBuf } },
    ],
  });
  submitCompute(dev, remapPipeline, bg, Math.ceil(totalPx / 256));
  const result = await readBackFloat32(dev, outputBuf, dataSize);
  releaseBuffer(paramBuf, 16, UNIFORM_DST);
  releaseBuffer(inputBuf, dataSize, STORAGE_DST);
  releaseBuffer(outputBuf, dataSize, STORAGE_SRC);
  releaseBuffer(mapXBuf, mapSize, STORAGE_DST);
  releaseBuffer(mapYBuf, mapSize, STORAGE_DST);
  const out = createImg(w, h, c);
  out.data.set(result);
  return out;
}

// ── GPU Batched Displace+Fade (all passes, single upload/download) ───────────

const DISPLACE_FADE_SHADER = /* wgsl */ `
struct Params { w: u32, h: u32, dmapC: u32, _p: u32, hScale: f32, vScale: f32, fadeA: f32, fadeB: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> dmap: array<f32>;

fn rc(v: i32, mx: i32) -> i32 {
  var r = v; if (r < 0) { r = -r; }
  if (r >= mx) { let p = mx * 2; r = r % p; if (r >= mx) { r = p - r - 1; } }
  return r;
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x; if (px >= params.w * params.h) { return; }
  let y = i32(px / params.w); let x = i32(px % params.w);
  let w = i32(params.w); let h = i32(params.h);
  let dmapC = params.dmapC;
  let hv = dmap[px * dmapC];
  let vv = dmap[px * dmapC + 1u];
  let fx = f32(x) + (hv - 128.0) / 128.0 * params.hScale;
  let fy = f32(y) + (vv - 128.0) / 128.0 * params.vScale;
  // Bilinear sample
  let x0 = i32(floor(fx)); let y0 = i32(floor(fy));
  let wx = fx - f32(x0); let wy = fy - f32(y0);
  let rx0 = rc(x0, w); let rx1 = rc(x0+1, w);
  let ry0 = rc(y0, h); let ry1 = rc(y0+1, h);
  let w00 = (1.0-wx)*(1.0-wy); let w10 = wx*(1.0-wy);
  let w01 = (1.0-wx)*wy; let w11 = wx*wy;
  let i00=u32(ry0*w+rx0)*3u; let i10=u32(ry0*w+rx1)*3u;
  let i01=u32(ry1*w+rx0)*3u; let i11=u32(ry1*w+rx1)*3u;
  let oi = px * 3u;
  // Displaced value
  let dr = input[i00]*w00+input[i10]*w10+input[i01]*w01+input[i11]*w11;
  let dg = input[i00+1u]*w00+input[i10+1u]*w10+input[i01+1u]*w01+input[i11+1u]*w11;
  let db = input[i00+2u]*w00+input[i10+2u]*w10+input[i01+2u]*w01+input[i11+2u]*w11;
  // Fade: out = displaced * fadeA + original * fadeB, clamped
  let origR = input[oi]; let origG = input[oi+1u]; let origB = input[oi+2u];
  output[oi]   = clamp(dr * params.fadeA + origR * params.fadeB, 0.0, 255.0);
  output[oi+1u]= clamp(dg * params.fadeA + origG * params.fadeB, 0.0, 255.0);
  output[oi+2u]= clamp(db * params.fadeA + origB * params.fadeB, 0.0, 255.0);
}`;

let displaceFadePipeline: GPUComputePipeline | null = null;

export async function gpuBatchDisplace(
  img: Img,
  dmap: Img,
  passes: [number, number, number][], // [hScale, vScale, fadePct]
): Promise<Img | null> {
  const dev = await getDevice();
  if (!dev) return null;
  const { w, h } = img;
  const totalPx = w * h;
  const dataSize = totalPx * 3 * 4;
  const dmapSize = totalPx * dmap.c * 4;

  if (!displaceFadePipeline) {
    displaceFadePipeline = dev.createComputePipeline({
      layout: "auto",
      compute: {
        module: dev.createShaderModule({ code: DISPLACE_FADE_SHADER }),
        entryPoint: "main",
      },
    });
  }

  const wgCount = Math.ceil(totalPx / 256);
  const midUsage = STORAGE_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;

  // Upload once: spray map + initial image
  const dmapBuf = allocAndWrite(dev, dmap.data as unknown as BufferSource, STORAGE_DST);
  const bufA = acquireBuffer(dev, dataSize, midUsage);
  const bufB = acquireBuffer(dev, dataSize, midUsage);
  dev.queue.writeBuffer(bufA, 0, img.data as unknown as BufferSource);

  // Encode all passes with ping-pong buffers
  const encoder = dev.createCommandEncoder();
  const paramBufs: GPUBuffer[] = [];

  for (let i = 0; i < passes.length; i++) {
    const [hs, vs, fp] = passes[i]!;
    const fadeA = fp / 100;
    const fadeB = 1 - fadeA;
    const paramData = new ArrayBuffer(32);
    new Uint32Array(paramData, 0, 4).set([w, h, dmap.c, 0]);
    new Float32Array(paramData, 16, 4).set([hs, vs, fadeA, fadeB]);
    const pBuf = allocAndWrite(dev, paramData, UNIFORM_DST);
    paramBufs.push(pBuf);

    const inBuf = i % 2 === 0 ? bufA : bufB;
    const outBuf = i % 2 === 0 ? bufB : bufA;

    const bg = dev.createBindGroup({
      layout: displaceFadePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: pBuf } },
        { binding: 1, resource: { buffer: inBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: dmapBuf } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(displaceFadePipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
    pass.end();
  }

  // Result is in the last-written buffer
  const resultBuf = passes.length % 2 === 0 ? bufA : bufB;
  dev.queue.submit([encoder.finish()]);

  const result = await readBackFloat32(dev, resultBuf, dataSize);

  releaseBuffer(dmapBuf, dmapSize, STORAGE_DST);
  releaseBuffer(bufA, dataSize, midUsage);
  releaseBuffer(bufB, dataSize, midUsage);
  for (const pb of paramBufs) releaseBuffer(pb, 32, UNIFORM_DST);

  const out = createImg(w, h, 3);
  out.data.set(result);
  return out;
}

// ── GPU Fused Distortion (ripple → roll → motionBlur → 6x displace+fade) ────
// Single upload, single readback. All intermediate data stays on GPU.

export async function gpuFusedDistortion(
  img: Img,
  sprayMap: Img,
  rippleRowDx: Float32Array,
  rippleColDy: Float32Array,
  rollDx: number,
  rollDy: number,
  motionKernelTaps: Float32Array,
  motionNumTaps: number,
  dispPasses: [number, number, number][],
): Promise<Img | null> {
  const dev = await getDevice();
  if (!dev) return null;
  const { w, h, c } = img;
  const totalPx = w * h;
  const dataSize = totalPx * c * 4;
  const wgCount = Math.ceil(totalPx / 256);
  const midUsage = STORAGE_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;

  // Ensure all pipelines exist
  if (!ripplePipeline) {
    ripplePipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: RIPPLE_SHADER }), entryPoint: "main" },
    });
  }
  if (!rollPipeline) {
    rollPipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: ROLL_SHADER }), entryPoint: "main" },
    });
  }
  if (!convolvePipeline) {
    convolvePipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: CONVOLVE_SHADER }), entryPoint: "main" },
    });
  }
  if (!displaceFadePipeline) {
    displaceFadePipeline = dev.createComputePipeline({
      layout: "auto",
      compute: {
        module: dev.createShaderModule({ code: DISPLACE_FADE_SHADER }),
        entryPoint: "main",
      },
    });
  }

  // Two ping-pong data buffers
  const bufA = acquireBuffer(dev, dataSize, midUsage);
  const bufB = acquireBuffer(dev, dataSize, midUsage);
  // Upload image into bufA
  dev.queue.writeBuffer(bufA, 0, img.data as unknown as BufferSource);

  // Upload small helper data
  const dxBuf = allocAndWrite(dev, rippleRowDx as unknown as BufferSource, STORAGE_DST);
  const dyBuf = allocAndWrite(dev, rippleColDy as unknown as BufferSource, STORAGE_DST);
  const tapsBuf = allocAndWrite(dev, motionKernelTaps as unknown as BufferSource, STORAGE_DST);
  const dmapSize = totalPx * sprayMap.c * 4;
  const dmapBuf = allocAndWrite(dev, sprayMap.data as unknown as BufferSource, STORAGE_DST);

  const encoder = dev.createCommandEncoder();
  let curIn = bufA;
  let curOut = bufB;
  const swap = () => {
    const tmp = curIn;
    curIn = curOut;
    curOut = tmp;
  };

  // Pass 1: Ripple (bufA → bufB)
  const rippleParams = allocAndWrite(dev, new Uint32Array([w, h, c, 0]), UNIFORM_DST);
  let pass = encoder.beginComputePass();
  pass.setPipeline(ripplePipeline);
  pass.setBindGroup(
    0,
    dev.createBindGroup({
      layout: ripplePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rippleParams } },
        { binding: 1, resource: { buffer: curIn } },
        { binding: 2, resource: { buffer: curOut } },
        { binding: 3, resource: { buffer: dxBuf } },
        { binding: 4, resource: { buffer: dyBuf } },
      ],
    }),
  );
  pass.dispatchWorkgroups(wgCount);
  pass.end();
  swap();

  // Pass 2: Roll (bufB → bufA)
  const rollParamData = new ArrayBuffer(32);
  new Uint32Array(rollParamData, 0, 4).set([w, h, c, 0]);
  new Int32Array(rollParamData, 16, 2).set([rollDx, rollDy]);
  const rollParams = allocAndWrite(dev, rollParamData, UNIFORM_DST);
  pass = encoder.beginComputePass();
  pass.setPipeline(rollPipeline);
  pass.setBindGroup(
    0,
    dev.createBindGroup({
      layout: rollPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rollParams } },
        { binding: 1, resource: { buffer: curIn } },
        { binding: 2, resource: { buffer: curOut } },
      ],
    }),
  );
  pass.dispatchWorkgroups(wgCount);
  pass.end();
  swap();

  // Pass 3: Motion blur convolution
  const convolveParams = allocAndWrite(dev, new Uint32Array([w, h, c, motionNumTaps]), UNIFORM_DST);
  pass = encoder.beginComputePass();
  pass.setPipeline(convolvePipeline);
  pass.setBindGroup(
    0,
    dev.createBindGroup({
      layout: convolvePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: convolveParams } },
        { binding: 1, resource: { buffer: curIn } },
        { binding: 2, resource: { buffer: curOut } },
        { binding: 3, resource: { buffer: tapsBuf } },
      ],
    }),
  );
  pass.dispatchWorkgroups(wgCount);
  pass.end();
  swap();

  // Passes 4-9: 6x displacement + fade
  const dispParamBufs: GPUBuffer[] = [];
  for (let i = 0; i < dispPasses.length; i++) {
    const [hs, vs, fp] = dispPasses[i]!;
    const fadeA = fp / 100;
    const fadeB = 1 - fadeA;
    const paramData = new ArrayBuffer(32);
    new Uint32Array(paramData, 0, 4).set([w, h, sprayMap.c, 0]);
    new Float32Array(paramData, 16, 4).set([hs, vs, fadeA, fadeB]);
    const pBuf = allocAndWrite(dev, paramData, UNIFORM_DST);
    dispParamBufs.push(pBuf);

    pass = encoder.beginComputePass();
    pass.setPipeline(displaceFadePipeline!);
    pass.setBindGroup(
      0,
      dev.createBindGroup({
        layout: displaceFadePipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: pBuf } },
          { binding: 1, resource: { buffer: curIn } },
          { binding: 2, resource: { buffer: curOut } },
          { binding: 3, resource: { buffer: dmapBuf } },
        ],
      }),
    );
    pass.dispatchWorkgroups(wgCount);
    pass.end();
    swap();
  }

  // Submit everything and read back final result
  dev.queue.submit([encoder.finish()]);
  const result = await readBackFloat32(dev, curIn, dataSize);

  // Cleanup
  releaseBuffer(bufA, dataSize, midUsage);
  releaseBuffer(bufB, dataSize, midUsage);
  releaseBuffer(dxBuf, rippleRowDx.byteLength, STORAGE_DST);
  releaseBuffer(dyBuf, rippleColDy.byteLength, STORAGE_DST);
  releaseBuffer(tapsBuf, motionKernelTaps.byteLength, STORAGE_DST);
  releaseBuffer(dmapBuf, dmapSize, STORAGE_DST);
  releaseBuffer(rippleParams, 16, UNIFORM_DST);
  releaseBuffer(rollParams, 32, UNIFORM_DST);
  releaseBuffer(convolveParams, 16, UNIFORM_DST);
  for (const pb of dispParamBufs) releaseBuffer(pb, 32, UNIFORM_DST);

  const out = createImg(w, h, c);
  out.data.set(result);
  return out;
}

// ── GPU Dilate / Erode ───────────────────────────────────────────────────────

const MORPH_SHADER = /* wgsl */ `
struct Params { w: u32, h: u32, numOffsets: u32, mode: u32 } // mode: 0=dilate, 1=erode
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> offsets: array<i32>;

fn rc(v: i32, mx: i32) -> i32 {
  var r = v; if (r < 0) { r = -r; }
  if (r >= mx) { let p = mx * 2; r = r % p; if (r >= mx) { r = p - r - 1; } }
  return r;
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x; if (px >= params.w * params.h) { return; }
  let y = i32(px / params.w); let x = i32(px % params.w);
  let w = i32(params.w); let h = i32(params.h);
  var val: f32;
  if (params.mode == 0u) { val = 0.0; } else { val = 255.0; }
  for (var i = 0u; i < params.numOffsets; i++) {
    let sy = rc(y + offsets[i*2u], h);
    let sx = rc(x + offsets[i*2u+1u], w);
    let v = input[u32(sy * w + sx)];
    if (params.mode == 0u) { val = max(val, v); } else { val = min(val, v); }
  }
  output[px] = val;
}`;

let morphPipeline: GPUComputePipeline | null = null;

async function gpuMorph(
  gray: Img,
  kernelSize: number,
  mask: Uint8Array,
  mode: number,
): Promise<Img | null> {
  const dev = await getDevice();
  if (!dev) return null;
  const { w, h } = gray;
  const r = (kernelSize - 1) >> 1;
  const offList: number[] = [];
  for (let ky = 0; ky < kernelSize; ky++)
    for (let kx = 0; kx < kernelSize; kx++) {
      if (mask[ky * kernelSize + kx]) offList.push(ky - r, kx - r);
    }
  const numOff = offList.length / 2;
  if (!morphPipeline) {
    morphPipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: MORPH_SHADER }), entryPoint: "main" },
    });
  }
  const totalPx = w * h;
  const dataSize = totalPx * 4;
  const offSize = Math.max(offList.length * 4, 16);
  const paramBuf = allocAndWrite(dev, new Uint32Array([w, h, numOff, mode]), UNIFORM_DST);
  const inputBuf = allocAndWrite(dev, gray.data as unknown as BufferSource, STORAGE_DST);
  const outputBuf = acquireBuffer(dev, dataSize, STORAGE_SRC);
  const offBuf = allocAndWrite(
    dev,
    new Int32Array(offList) as unknown as BufferSource,
    STORAGE_DST,
  );
  const bg = dev.createBindGroup({
    layout: morphPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: offBuf } },
    ],
  });
  submitCompute(dev, morphPipeline, bg, Math.ceil(totalPx / 256));
  const result = await readBackFloat32(dev, outputBuf, dataSize);
  releaseBuffer(paramBuf, 16, UNIFORM_DST);
  releaseBuffer(inputBuf, dataSize, STORAGE_DST);
  releaseBuffer(outputBuf, dataSize, STORAGE_SRC);
  releaseBuffer(offBuf, offSize, STORAGE_DST);
  const out = createImg(w, h, 1);
  out.data.set(result);
  return out;
}

export async function gpuDilate(
  gray: Img,
  kernelSize: number,
  mask: Uint8Array,
): Promise<Img | null> {
  return gpuMorph(gray, kernelSize, mask, 0);
}

export async function gpuErode(
  gray: Img,
  kernelSize: number,
  mask: Uint8Array,
): Promise<Img | null> {
  return gpuMorph(gray, kernelSize, mask, 1);
}

// ── GPU Separable Gaussian Blur ──────────────────────────────────────────────

const BLUR1D_SHADER = /* wgsl */ `
struct Params { w: u32, h: u32, c: u32, radius: u32, dir: u32, _p1: u32, _p2: u32, _p3: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;

fn rc(v: i32, mx: i32) -> i32 {
  var r = v; if (r < 0) { r = -r; }
  if (r >= mx) { let p = mx * 2; r = r % p; if (r >= mx) { r = p - r - 1; } }
  return r;
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x; if (px >= params.w * params.h) { return; }
  let y = i32(px / params.w); let x = i32(px % params.w);
  let w = i32(params.w); let h = i32(params.h);
  let r = i32(params.radius);
  let c = params.c;
  if (c == 1u) {
    var s: f32 = 0.0;
    for (var i = -r; i <= r; i++) {
      var sy: i32; var sx: i32;
      if (params.dir == 0u) { sy = y; sx = rc(x + i, w); }
      else { sy = rc(y + i, h); sx = x; }
      s += input[u32(sy * w + sx)] * weights[u32(i + r)];
    }
    output[px] = s;
  } else {
    var s0: f32 = 0.0; var s1: f32 = 0.0; var s2: f32 = 0.0;
    for (var i = -r; i <= r; i++) {
      var sy: i32; var sx: i32;
      if (params.dir == 0u) { sy = y; sx = rc(x + i, w); }
      else { sy = rc(y + i, h); sx = x; }
      let idx = u32(sy * w + sx) * 3u;
      let wt = weights[u32(i + r)];
      s0 += input[idx] * wt; s1 += input[idx+1u] * wt; s2 += input[idx+2u] * wt;
    }
    let oi = px * 3u; output[oi] = s0; output[oi+1u] = s1; output[oi+2u] = s2;
  }
}`;

let blur1dPipeline: GPUComputePipeline | null = null;

function makeGaussianKernel1D(sigma: number): Float32Array {
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const size = radius * 2 + 1;
  const kernel = new Float32Array(size);
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = i - radius;
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += kernel[i]!;
  }
  for (let i = 0; i < size; i++) kernel[i] /= sum;
  return kernel;
}

export async function gpuGaussianBlur(img: Img, sigma: number): Promise<Img | null> {
  const dev = await getDevice();
  if (!dev) return null;
  const { w, h, c } = img;
  const kernel = makeGaussianKernel1D(sigma);
  const radius = (kernel.length - 1) / 2;
  if (!blur1dPipeline) {
    blur1dPipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: BLUR1D_SHADER }), entryPoint: "main" },
    });
  }
  const totalPx = w * h;
  const dataSize = totalPx * c * 4;
  const wgCount = Math.ceil(totalPx / 256);

  // Horizontal pass
  const paramH = allocAndWrite(dev, new Uint32Array([w, h, c, radius, 0, 0, 0, 0]), UNIFORM_DST);
  const inputBuf = allocAndWrite(dev, img.data as unknown as BufferSource, STORAGE_DST);
  const midBuf = acquireBuffer(
    dev,
    dataSize,
    STORAGE_SRC | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const wBuf = allocAndWrite(dev, kernel as unknown as BufferSource, STORAGE_DST);

  const bgH = dev.createBindGroup({
    layout: blur1dPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramH } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: midBuf } },
      { binding: 3, resource: { buffer: wBuf } },
    ],
  });

  // Vertical pass
  const paramV = allocAndWrite(dev, new Uint32Array([w, h, c, radius, 1, 0, 0, 0]), UNIFORM_DST);
  const outputBuf = acquireBuffer(dev, dataSize, STORAGE_SRC);

  const bgV = dev.createBindGroup({
    layout: blur1dPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramV } },
      { binding: 1, resource: { buffer: midBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: wBuf } },
    ],
  });

  // Submit both passes in one command
  const encoder = dev.createCommandEncoder();
  let pass = encoder.beginComputePass();
  pass.setPipeline(blur1dPipeline);
  pass.setBindGroup(0, bgH);
  pass.dispatchWorkgroups(wgCount);
  pass.end();
  pass = encoder.beginComputePass();
  pass.setPipeline(blur1dPipeline);
  pass.setBindGroup(0, bgV);
  pass.dispatchWorkgroups(wgCount);
  pass.end();
  dev.queue.submit([encoder.finish()]);

  const result = await readBackFloat32(dev, outputBuf, dataSize);
  const midUsage = STORAGE_SRC | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  releaseBuffer(paramH, 32, UNIFORM_DST);
  releaseBuffer(paramV, 32, UNIFORM_DST);
  releaseBuffer(inputBuf, dataSize, STORAGE_DST);
  releaseBuffer(midBuf, dataSize, midUsage);
  releaseBuffer(outputBuf, dataSize, STORAGE_SRC);
  releaseBuffer(wBuf, kernel.byteLength, STORAGE_DST);
  const out = createImg(w, h, c);
  out.data.set(result);
  return out;
}

// ── GPU Paint Grain (fused grayscale + grain) ────────────────────────────────

const GRAIN_SHADER = /* wgsl */ `
struct Params { w: u32, h: u32, darkFactor: f32, _p: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> rng: array<f32>; // pre-generated random [0,256)

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x; if (px >= params.w * params.h) { return; }
  let oi = px * 3u;
  let r = input[oi]; let g = input[oi+1u]; let b = input[oi+2u];
  let gray = 0.299 * r + 0.587 * g + 0.114 * b;
  let rand = rng[px];
  let mask = select(0.0, 1.0, rand < gray);
  let df = params.darkFactor;
  output[oi]   = r * mask + r * df * (1.0 - mask);
  output[oi+1u]= g * mask + g * df * (1.0 - mask);
  output[oi+2u]= b * mask + b * df * (1.0 - mask);
}`;

let grainPipeline: GPUComputePipeline | null = null;

export async function gpuPaintGrain(
  img: Img,
  rngValues: Float32Array,
  darkFactor: number,
): Promise<Img | null> {
  const dev = await getDevice();
  if (!dev) return null;
  const { w, h } = img;
  if (!grainPipeline) {
    grainPipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: GRAIN_SHADER }), entryPoint: "main" },
    });
  }
  const totalPx = w * h;
  const dataSize = totalPx * 3 * 4;
  const rngSize = totalPx * 4;

  // Pack darkFactor as float in Uint32 uniform
  const paramData = new ArrayBuffer(16);
  new Uint32Array(paramData, 0, 2).set([w, h]);
  new Float32Array(paramData, 8, 1).set([darkFactor]);
  new Uint32Array(paramData, 12, 1).set([0]);

  const paramBuf = allocAndWrite(dev, paramData, UNIFORM_DST);
  const inputBuf = allocAndWrite(dev, img.data as unknown as BufferSource, STORAGE_DST);
  const outputBuf = acquireBuffer(dev, dataSize, STORAGE_SRC);
  const rngBuf = allocAndWrite(dev, rngValues as unknown as BufferSource, STORAGE_DST);

  const bg = dev.createBindGroup({
    layout: grainPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: rngBuf } },
    ],
  });
  submitCompute(dev, grainPipeline, bg, Math.ceil(totalPx / 256));
  const result = await readBackFloat32(dev, outputBuf, dataSize);
  releaseBuffer(paramBuf, 16, UNIFORM_DST);
  releaseBuffer(inputBuf, dataSize, STORAGE_DST);
  releaseBuffer(outputBuf, dataSize, STORAGE_SRC);
  releaseBuffer(rngBuf, rngSize, STORAGE_DST);
  const out = createImg(w, h, 3);
  out.data.set(result);
  return out;
}

// ── GPU Sharpen (fused: two blurs + unsharp mask, single readback) ───────────

export async function gpuSharpen(img: Img, sigma1: number, sigma2: number): Promise<Img | null> {
  const dev = await getDevice();
  if (!dev) return null;
  const { w, h, c } = img;
  const k1 = makeGaussianKernel1D(sigma1);
  const k2 = makeGaussianKernel1D(sigma2);
  const r1 = (k1.length - 1) / 2;
  const r2 = (k2.length - 1) / 2;
  if (!blur1dPipeline) {
    blur1dPipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: BLUR1D_SHADER }), entryPoint: "main" },
    });
  }
  const totalPx = w * h;
  const dataSize = totalPx * c * 4;
  const wgCount = Math.ceil(totalPx / 256);
  const midUsage = STORAGE_SRC | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

  const inputBuf = allocAndWrite(dev, img.data as unknown as BufferSource, STORAGE_DST);
  const buf1 = acquireBuffer(dev, dataSize, midUsage); // after sigma1 h-pass
  const buf2 = acquireBuffer(dev, dataSize, midUsage); // blurred04 (after sigma1 v-pass)
  const buf3 = acquireBuffer(dev, dataSize, midUsage); // after sigma2 h-pass
  const buf4 = acquireBuffer(dev, dataSize, STORAGE_SRC); // blurred2 (after sigma2 v-pass)
  const w1Buf = allocAndWrite(dev, k1 as unknown as BufferSource, STORAGE_DST);
  const w2Buf = allocAndWrite(dev, k2 as unknown as BufferSource, STORAGE_DST);

  const mkParams = (radius: number, dir: number) =>
    allocAndWrite(dev, new Uint32Array([w, h, c, radius, dir, 0, 0, 0]), UNIFORM_DST);
  const p1h = mkParams(r1, 0);
  const p1v = mkParams(r1, 1);
  const p2h = mkParams(r2, 0);
  const p2v = mkParams(r2, 1);

  const mkBg = (pBuf: GPUBuffer, inBuf: GPUBuffer, outBuf: GPUBuffer, wBuf: GPUBuffer) =>
    dev.createBindGroup({
      layout: blur1dPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: pBuf } },
        { binding: 1, resource: { buffer: inBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: wBuf } },
      ],
    });

  // Encode all 4 blur passes in one command buffer
  const encoder = dev.createCommandEncoder();
  const passes = [
    mkBg(p1h, inputBuf, buf1, w1Buf), // sigma1 horizontal
    mkBg(p1v, buf1, buf2, w1Buf), // sigma1 vertical -> blurred04
    mkBg(p2h, buf2, buf3, w2Buf), // sigma2 horizontal (on blurred04)
    mkBg(p2v, buf3, buf4, w2Buf), // sigma2 vertical -> blurred2
  ];
  for (const bg of passes) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(blur1dPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
    pass.end();
  }
  // Copy both results in the same command buffer as the blur passes
  const readBuf1 = dev.createBuffer({ size: dataSize, usage: READ_DST });
  const readBuf2 = dev.createBuffer({ size: dataSize, usage: READ_DST });
  encoder.copyBufferToBuffer(buf2, 0, readBuf1, 0, dataSize);
  encoder.copyBufferToBuffer(buf4, 0, readBuf2, 0, dataSize);
  dev.queue.submit([encoder.finish()]);

  // Read back both results
  await Promise.all([readBuf1.mapAsync(GPUMapMode.READ), readBuf2.mapAsync(GPUMapMode.READ)]);
  const m1 = new Float32Array(readBuf1.getMappedRange());
  const r04 = new Float32Array(m1.length);
  r04.set(m1);
  readBuf1.unmap();
  readBuf1.destroy();
  const m2 = new Float32Array(readBuf2.getMappedRange());
  const r2data = new Float32Array(m2.length);
  r2data.set(m2);
  readBuf2.unmap();
  readBuf2.destroy();

  // Cleanup
  for (const b of [inputBuf, buf1, buf3]) releaseBuffer(b, dataSize, midUsage);
  releaseBuffer(buf2, dataSize, midUsage);
  releaseBuffer(buf4, dataSize, STORAGE_SRC);
  releaseBuffer(w1Buf, k1.byteLength, STORAGE_DST);
  releaseBuffer(w2Buf, k2.byteLength, STORAGE_DST);
  for (const b of [p1h, p1v, p2h, p2v]) releaseBuffer(b, 32, UNIFORM_DST);

  // Unsharp mask: clamp(blurred04 + (blurred04 - blurred2))
  const out = createImg(w, h, c);
  for (let i = 0; i < r04.length; i++) {
    const v = r04[i]! + (r04[i]! - r2data[i]!);
    out.data[i] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  return out;
}

let warmedUp = false;

/** Pre-compile all GPU pipelines AND run a tiny dummy dispatch to trigger
 *  driver-level JIT so the first real dispatch doesn't pay warm-up cost */
export async function warmUpGpu(): Promise<void> {
  if (warmedUp) return;
  const dev = await getDevice();
  if (!dev) return;

  // Create pipelines
  if (!convolvePipeline) {
    convolvePipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: CONVOLVE_SHADER }), entryPoint: "main" },
    });
  }
  if (!remapPipeline) {
    remapPipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: REMAP_SHADER }), entryPoint: "main" },
    });
  }
  if (!morphPipeline) {
    morphPipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: MORPH_SHADER }), entryPoint: "main" },
    });
  }
  if (!blur1dPipeline) {
    blur1dPipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: BLUR1D_SHADER }), entryPoint: "main" },
    });
  }
  if (!grainPipeline) {
    grainPipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: GRAIN_SHADER }), entryPoint: "main" },
    });
  }

  // Run a tiny 1-pixel dispatch through each pipeline to trigger driver JIT
  const dummyBuf = dev.createBuffer({ size: 16, usage: STORAGE_DST | GPUBufferUsage.COPY_SRC });
  const dummyOut = dev.createBuffer({ size: 16, usage: STORAGE_SRC | GPUBufferUsage.STORAGE });
  const dummyUni = dev.createBuffer({ size: 32, usage: UNIFORM_DST });

  const encoder = dev.createCommandEncoder();

  // Warm convolve (needs: uniform, input, output, taps)
  dev.queue.writeBuffer(dummyUni, 0, new Uint32Array([1, 1, 1, 1]));
  dev.queue.writeBuffer(dummyBuf, 0, new Float32Array([0, 0, 0, 1])); // 1 tap
  let pass = encoder.beginComputePass();
  pass.setPipeline(convolvePipeline);
  pass.setBindGroup(
    0,
    dev.createBindGroup({
      layout: convolvePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dummyUni } },
        { binding: 1, resource: { buffer: dummyBuf } },
        { binding: 2, resource: { buffer: dummyOut } },
        { binding: 3, resource: { buffer: dummyBuf } },
      ],
    }),
  );
  pass.dispatchWorkgroups(1);
  pass.end();

  // Warm remap (needs: uniform, input, output, mapX, mapY)
  dev.queue.writeBuffer(dummyUni, 0, new Uint32Array([1, 1, 1, 0, 0, 0, 0, 0]));
  pass = encoder.beginComputePass();
  pass.setPipeline(remapPipeline);
  pass.setBindGroup(
    0,
    dev.createBindGroup({
      layout: remapPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dummyUni } },
        { binding: 1, resource: { buffer: dummyBuf } },
        { binding: 2, resource: { buffer: dummyOut } },
        { binding: 3, resource: { buffer: dummyBuf } },
        { binding: 4, resource: { buffer: dummyBuf } },
      ],
    }),
  );
  pass.dispatchWorkgroups(1);
  pass.end();

  // Warm morph (needs: uniform, input, output, offsets)
  dev.queue.writeBuffer(dummyUni, 0, new Uint32Array([1, 1, 1, 0]));
  pass = encoder.beginComputePass();
  pass.setPipeline(morphPipeline);
  pass.setBindGroup(
    0,
    dev.createBindGroup({
      layout: morphPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dummyUni } },
        { binding: 1, resource: { buffer: dummyBuf } },
        { binding: 2, resource: { buffer: dummyOut } },
        { binding: 3, resource: { buffer: dummyBuf } },
      ],
    }),
  );
  pass.dispatchWorkgroups(1);
  pass.end();

  // Warm blur1d (needs: uniform, input, output, weights)
  dev.queue.writeBuffer(dummyUni, 0, new Uint32Array([1, 1, 1, 0, 0, 0, 0, 0]));
  pass = encoder.beginComputePass();
  pass.setPipeline(blur1dPipeline);
  pass.setBindGroup(
    0,
    dev.createBindGroup({
      layout: blur1dPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dummyUni } },
        { binding: 1, resource: { buffer: dummyBuf } },
        { binding: 2, resource: { buffer: dummyOut } },
        { binding: 3, resource: { buffer: dummyBuf } },
      ],
    }),
  );
  pass.dispatchWorkgroups(1);
  pass.end();

  // Warm roll
  if (!rollPipeline) {
    rollPipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: ROLL_SHADER }), entryPoint: "main" },
    });
  }
  dev.queue.writeBuffer(dummyUni, 0, new ArrayBuffer(32));
  pass = encoder.beginComputePass();
  pass.setPipeline(rollPipeline);
  pass.setBindGroup(
    0,
    dev.createBindGroup({
      layout: rollPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dummyUni } },
        { binding: 1, resource: { buffer: dummyBuf } },
        { binding: 2, resource: { buffer: dummyOut } },
      ],
    }),
  );
  pass.dispatchWorkgroups(1);
  pass.end();

  // Warm ripple
  if (!ripplePipeline) {
    ripplePipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module: dev.createShaderModule({ code: RIPPLE_SHADER }), entryPoint: "main" },
    });
  }
  pass = encoder.beginComputePass();
  pass.setPipeline(ripplePipeline);
  pass.setBindGroup(
    0,
    dev.createBindGroup({
      layout: ripplePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dummyUni } },
        { binding: 1, resource: { buffer: dummyBuf } },
        { binding: 2, resource: { buffer: dummyOut } },
        { binding: 3, resource: { buffer: dummyBuf } },
        { binding: 4, resource: { buffer: dummyBuf } },
      ],
    }),
  );
  pass.dispatchWorkgroups(1);
  pass.end();

  // Warm displaceFade
  if (!displaceFadePipeline) {
    displaceFadePipeline = dev.createComputePipeline({
      layout: "auto",
      compute: {
        module: dev.createShaderModule({ code: DISPLACE_FADE_SHADER }),
        entryPoint: "main",
      },
    });
  }
  const dfUni = dev.createBuffer({ size: 32, usage: UNIFORM_DST });
  const dfParamData = new ArrayBuffer(32);
  new Uint32Array(dfParamData, 0, 4).set([1, 1, 3, 0]);
  new Float32Array(dfParamData, 16, 4).set([0, 0, 0.75, 0.25]);
  dev.queue.writeBuffer(dfUni, 0, dfParamData);
  pass = encoder.beginComputePass();
  pass.setPipeline(displaceFadePipeline);
  pass.setBindGroup(
    0,
    dev.createBindGroup({
      layout: displaceFadePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dfUni } },
        { binding: 1, resource: { buffer: dummyBuf } },
        { binding: 2, resource: { buffer: dummyOut } },
        { binding: 3, resource: { buffer: dummyBuf } },
      ],
    }),
  );
  pass.dispatchWorkgroups(1);
  pass.end();

  // Warm grain (needs: uniform, input[rgb], output[rgb], rng)
  dev.queue.writeBuffer(dummyUni, 0, new Uint32Array([1, 1, 0, 0]));
  new Float32Array(new Uint32Array([1, 1, 0, 0]).buffer); // just for the darkFactor
  const grainUni = dev.createBuffer({ size: 16, usage: UNIFORM_DST });
  const grainParamData = new ArrayBuffer(16);
  new Uint32Array(grainParamData, 0, 2).set([1, 1]);
  new Float32Array(grainParamData, 8, 1).set([0.85]);
  dev.queue.writeBuffer(grainUni, 0, grainParamData);
  const dummyRgbBuf = dev.createBuffer({ size: 16, usage: STORAGE_DST | GPUBufferUsage.COPY_SRC });
  const dummyRgbOut = dev.createBuffer({ size: 16, usage: STORAGE_SRC | GPUBufferUsage.STORAGE });
  pass = encoder.beginComputePass();
  pass.setPipeline(grainPipeline);
  pass.setBindGroup(
    0,
    dev.createBindGroup({
      layout: grainPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: grainUni } },
        { binding: 1, resource: { buffer: dummyRgbBuf } },
        { binding: 2, resource: { buffer: dummyRgbOut } },
        { binding: 3, resource: { buffer: dummyBuf } },
      ],
    }),
  );
  pass.dispatchWorkgroups(1);
  pass.end();

  dev.queue.submit([encoder.finish()]);

  // Wait for all dummy dispatches to complete
  await dev.queue.onSubmittedWorkDone();

  // Cleanup dummy buffers
  dummyBuf.destroy();
  dummyOut.destroy();
  dummyUni.destroy();
  grainUni.destroy();
  dummyRgbBuf.destroy();
  dummyRgbOut.destroy();
  dfUni.destroy();

  warmedUp = true;
}
