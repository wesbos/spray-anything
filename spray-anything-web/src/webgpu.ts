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

export async function isWebGPUAvailable(): Promise<boolean> {
  await getDevice();
  return gpuAvailable === true;
}

// ── GPU Convolution (sparse) ─────────────────────────────────────────────────

const CONVOLVE_SHADER = /* wgsl */ `
struct Params {
  w: u32,
  h: u32,
  c: u32,
  numTaps: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
// Sparse kernel: [dy, dx, weight] triples packed as f32
@group(0) @binding(3) var<storage, read> taps: array<f32>;

fn reflectCoord(v: i32, max: i32) -> i32 {
  var r = v;
  if (r < 0) { r = -r; }
  if (r >= max) {
    let period = max * 2;
    r = r % period;
    if (r >= max) { r = period - r - 1; }
  }
  return r;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = gid.x;
  let total = params.w * params.h;
  if (pixel >= total) { return; }

  let y = i32(pixel / params.w);
  let x = i32(pixel % params.w);
  let w = i32(params.w);
  let h = i32(params.h);
  let c = params.c;
  let numTaps = params.numTaps;

  if (c == 1u) {
    var sum: f32 = 0.0;
    for (var i = 0u; i < numTaps; i++) {
      let dy = i32(taps[i * 3u]);
      let dx = i32(taps[i * 3u + 1u]);
      let wt = taps[i * 3u + 2u];
      let sy = reflectCoord(y + dy, h);
      let sx = reflectCoord(x + dx, w);
      sum += input[u32(sy * w + sx)] * wt;
    }
    output[pixel] = sum;
  } else {
    // c == 3
    var s0: f32 = 0.0;
    var s1: f32 = 0.0;
    var s2: f32 = 0.0;
    for (var i = 0u; i < numTaps; i++) {
      let dy = i32(taps[i * 3u]);
      let dx = i32(taps[i * 3u + 1u]);
      let wt = taps[i * 3u + 2u];
      let sy = reflectCoord(y + dy, h);
      let sx = reflectCoord(x + dx, w);
      let idx = u32(sy * w + sx) * 3u;
      s0 += input[idx] * wt;
      s1 += input[idx + 1u] * wt;
      s2 += input[idx + 2u] * wt;
    }
    let oi = pixel * 3u;
    output[oi] = s0;
    output[oi + 1u] = s1;
    output[oi + 2u] = s2;
  }
}
`;

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
  const khh = (kh - 1) >> 1;
  const kwh = (kw - 1) >> 1;

  // Build sparse taps: [dy, dx, weight] triples
  const tapsList: number[] = [];
  for (let ky = 0; ky < kh; ky++) {
    for (let kx = 0; kx < kw; kx++) {
      const v = kernel[ky * kw + kx]!;
      if (v !== 0) {
        tapsList.push(ky - khh, kx - kwh, v);
      }
    }
  }
  const numTaps = tapsList.length / 3;
  const tapsData = new Float32Array(tapsList);

  // Create pipeline (cached)
  if (!convolvePipeline) {
    const module = dev.createShaderModule({ code: CONVOLVE_SHADER });
    convolvePipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  const totalPixels = w * h;
  const dataSize = totalPixels * c * 4;

  const paramBuf = dev.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(paramBuf, 0, new Uint32Array([w, h, c, numTaps]));

  const inputBuf = dev.createBuffer({
    size: dataSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(inputBuf, 0, img.data as unknown as BufferSource);

  const outputBuf = dev.createBuffer({
    size: dataSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const tapsBuf = dev.createBuffer({
    size: Math.max(tapsData.byteLength, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(tapsBuf, 0, tapsData);

  const readBuf = dev.createBuffer({
    size: dataSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = dev.createBindGroup({
    layout: convolvePipeline!.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: tapsBuf } },
    ],
  });

  const encoder = dev.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(convolvePipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(totalPixels / 256));
  pass.end();
  encoder.copyBufferToBuffer(outputBuf, 0, readBuf, 0, dataSize);
  dev.queue.submit([encoder.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const out = createImg(w, h, c);
  new Float32Array(out.data.buffer).set(new Float32Array(readBuf.getMappedRange()));
  readBuf.unmap();

  paramBuf.destroy();
  inputBuf.destroy();
  outputBuf.destroy();
  tapsBuf.destroy();
  readBuf.destroy();

  return out;
}

// ── GPU Remap (bilinear) ─────────────────────────────────────────────────────

const REMAP_SHADER = /* wgsl */ `
struct Params {
  w: u32,
  h: u32,
  c: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> mapX: array<f32>;
@group(0) @binding(4) var<storage, read> mapY: array<f32>;

fn reflectCoord(v: i32, max: i32) -> i32 {
  var r = v;
  if (r < 0) { r = -r; }
  if (r >= max) {
    let period = max * 2;
    r = r % period;
    if (r >= max) { r = period - r - 1; }
  }
  return r;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = gid.x;
  let total = params.w * params.h;
  if (pixel >= total) { return; }

  let w = i32(params.w);
  let h = i32(params.h);
  let c = params.c;

  let fx = mapX[pixel];
  let fy = mapY[pixel];
  let x0 = i32(floor(fx));
  let y0 = i32(floor(fy));
  let wx = fx - f32(x0);
  let wy = fy - f32(y0);

  let rx0 = reflectCoord(x0, w);
  let rx1 = reflectCoord(x0 + 1, w);
  let ry0 = reflectCoord(y0, h);
  let ry1 = reflectCoord(y0 + 1, h);

  let w00 = (1.0 - wx) * (1.0 - wy);
  let w10 = wx * (1.0 - wy);
  let w01 = (1.0 - wx) * wy;
  let w11 = wx * wy;

  if (c == 1u) {
    let v = f32(input[u32(ry0 * w + rx0)]) * w00
          + f32(input[u32(ry0 * w + rx1)]) * w10
          + f32(input[u32(ry1 * w + rx0)]) * w01
          + f32(input[u32(ry1 * w + rx1)]) * w11;
    output[pixel] = v;
  } else {
    let i00 = u32(ry0 * w + rx0) * 3u;
    let i10 = u32(ry0 * w + rx1) * 3u;
    let i01 = u32(ry1 * w + rx0) * 3u;
    let i11 = u32(ry1 * w + rx1) * 3u;
    let oi = pixel * 3u;
    output[oi]     = input[i00] * w00 + input[i10] * w10 + input[i01] * w01 + input[i11] * w11;
    output[oi + 1u] = input[i00+1u] * w00 + input[i10+1u] * w10 + input[i01+1u] * w01 + input[i11+1u] * w11;
    output[oi + 2u] = input[i00+2u] * w00 + input[i10+2u] * w10 + input[i01+2u] * w01 + input[i11+2u] * w11;
  }
}
`;

let remapPipeline: GPUComputePipeline | null = null;

export async function gpuRemap(
  img: Img,
  mapXData: Float32Array,
  mapYData: Float32Array,
): Promise<Img | null> {
  const dev = await getDevice();
  if (!dev) return null;

  const { w, h, c } = img;
  const totalPixels = w * h;
  const dataSize = totalPixels * c * 4;
  const mapSize = totalPixels * 4;

  if (!remapPipeline) {
    const module = dev.createShaderModule({ code: REMAP_SHADER });
    remapPipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  const paramBuf = dev.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(paramBuf, 0, new Uint32Array([w, h, c, 0]));

  const inputBuf = dev.createBuffer({
    size: dataSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(inputBuf, 0, img.data as unknown as BufferSource);

  const outputBuf = dev.createBuffer({
    size: dataSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const mapXBuf = dev.createBuffer({
    size: mapSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(mapXBuf, 0, mapXData as unknown as BufferSource);

  const mapYBuf = dev.createBuffer({
    size: mapSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(mapYBuf, 0, mapYData as unknown as BufferSource);

  const readBuf = dev.createBuffer({
    size: dataSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = dev.createBindGroup({
    layout: remapPipeline!.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: mapXBuf } },
      { binding: 4, resource: { buffer: mapYBuf } },
    ],
  });

  const encoder = dev.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(remapPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(totalPixels / 256));
  pass.end();
  encoder.copyBufferToBuffer(outputBuf, 0, readBuf, 0, dataSize);
  dev.queue.submit([encoder.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const out = createImg(w, h, c);
  new Float32Array(out.data.buffer).set(new Float32Array(readBuf.getMappedRange()));
  readBuf.unmap();

  paramBuf.destroy();
  inputBuf.destroy();
  outputBuf.destroy();
  mapXBuf.destroy();
  mapYBuf.destroy();
  readBuf.destroy();

  return out;
}

// ── GPU Dilate (elliptical structuring element) ──────────────────────────────

const DILATE_SHADER = /* wgsl */ `
struct Params {
  w: u32,
  h: u32,
  kernelSize: u32,
  numOffsets: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> offsets: array<i32>;

fn reflectCoord(v: i32, max: i32) -> i32 {
  var r = v;
  if (r < 0) { r = -r; }
  if (r >= max) {
    let period = max * 2;
    r = r % period;
    if (r >= max) { r = period - r - 1; }
  }
  return r;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = gid.x;
  let total = params.w * params.h;
  if (pixel >= total) { return; }

  let y = i32(pixel / params.w);
  let x = i32(pixel % params.w);
  let w = i32(params.w);
  let h = i32(params.h);
  let numOffsets = params.numOffsets;

  var maxVal: f32 = 0.0;
  for (var i = 0u; i < numOffsets; i++) {
    let dy = offsets[i * 2u];
    let dx = offsets[i * 2u + 1u];
    let sy = reflectCoord(y + dy, h);
    let sx = reflectCoord(x + dx, w);
    let v = input[u32(sy * w + sx)];
    maxVal = max(maxVal, v);
  }
  output[pixel] = maxVal;
}
`;

let dilatePipeline: GPUComputePipeline | null = null;

export async function gpuDilate(
  gray: Img,
  kernelSize: number,
  mask: Uint8Array,
): Promise<Img | null> {
  const dev = await getDevice();
  if (!dev) return null;

  const { w, h } = gray;
  const r = (kernelSize - 1) >> 1;

  // Precompute active offsets from mask
  const offsetsList: number[] = [];
  for (let ky = 0; ky < kernelSize; ky++) {
    for (let kx = 0; kx < kernelSize; kx++) {
      if (mask[ky * kernelSize + kx]) {
        offsetsList.push(ky - r, kx - r);
      }
    }
  }
  const numOffsets = offsetsList.length / 2;
  const offsetsData = new Int32Array(offsetsList);

  if (!dilatePipeline) {
    const module = dev.createShaderModule({ code: DILATE_SHADER });
    dilatePipeline = dev.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  const totalPixels = w * h;
  const dataSize = totalPixels * 4;

  const paramBuf = dev.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(paramBuf, 0, new Uint32Array([w, h, kernelSize, numOffsets]));

  const inputBuf = dev.createBuffer({
    size: dataSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(inputBuf, 0, gray.data as unknown as BufferSource);

  const outputBuf = dev.createBuffer({
    size: dataSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const offsetsBuf = dev.createBuffer({
    size: Math.max(offsetsData.byteLength, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(offsetsBuf, 0, offsetsData as unknown as BufferSource);

  const readBuf = dev.createBuffer({
    size: dataSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = dev.createBindGroup({
    layout: dilatePipeline!.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: offsetsBuf } },
    ],
  });

  const encoder = dev.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(dilatePipeline!);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(totalPixels / 256));
  pass.end();
  encoder.copyBufferToBuffer(outputBuf, 0, readBuf, 0, dataSize);
  dev.queue.submit([encoder.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const out = createImg(w, h, 1);
  new Float32Array(out.data.buffer).set(new Float32Array(readBuf.getMappedRange()));
  readBuf.unmap();

  paramBuf.destroy();
  inputBuf.destroy();
  outputBuf.destroy();
  offsetsBuf.destroy();
  readBuf.destroy();

  return out;
}
