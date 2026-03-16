import { type Img, createImg, reflectCoord } from "./image.ts";
import { SeededRNG } from "./rng.ts";
import { gpuConvolve2d, gpuRemap, gpuDilate, gpuErode, gpuRipple } from "./webgpu.ts";

function sampleBilinear(img: Img, fx: number, fy: number): number[] {
  const { w, h, c } = img;
  const x0 = Math.floor(fx),
    y0 = Math.floor(fy);
  const x1 = x0 + 1,
    y1 = y0 + 1;
  const wx = fx - x0,
    wy = fy - y0;

  const rx0 = reflectCoord(x0, w),
    rx1 = reflectCoord(x1, w);
  const ry0 = reflectCoord(y0, h),
    ry1 = reflectCoord(y1, h);

  const i00 = (ry0 * w + rx0) * c;
  const i10 = (ry0 * w + rx1) * c;
  const i01 = (ry1 * w + rx0) * c;
  const i11 = (ry1 * w + rx1) * c;

  const result = Array.from<number>({ length: c });
  const w00 = (1 - wx) * (1 - wy);
  const w10 = wx * (1 - wy);
  const w01 = (1 - wx) * wy;
  const w11 = wx * wy;
  for (let ch = 0; ch < c; ch++) {
    result[ch] =
      img.data[i00 + ch]! * w00 +
      img.data[i10 + ch]! * w10 +
      img.data[i01 + ch]! * w01 +
      img.data[i11 + ch]! * w11;
  }
  return result;
}

export function remapCpu(img: Img, mapX: Float32Array, mapY: Float32Array): Img {
  const out = createImg(img.w, img.h, img.c);
  const { w, h, c, data } = img;
  if (c === 3) {
    // Fast path: inline bilinear for 3-channel, no per-pixel allocation
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        const fx = mapX[idx]!;
        const fy = mapY[idx]!;
        const x0 = Math.floor(fx),
          y0 = Math.floor(fy);
        const wx = fx - x0,
          wy = fy - y0;

        const rx0 = reflectCoord(x0, w),
          rx1 = reflectCoord(x0 + 1, w);
        const ry0 = reflectCoord(y0, h),
          ry1 = reflectCoord(y0 + 1, h);

        const i00 = (ry0 * w + rx0) * 3;
        const i10 = (ry0 * w + rx1) * 3;
        const i01 = (ry1 * w + rx0) * 3;
        const i11 = (ry1 * w + rx1) * 3;

        const w00 = (1 - wx) * (1 - wy);
        const w10 = wx * (1 - wy);
        const w01 = (1 - wx) * wy;
        const w11 = wx * wy;

        const oi = idx * 3;
        out.data[oi] = data[i00]! * w00 + data[i10]! * w10 + data[i01]! * w01 + data[i11]! * w11;
        out.data[oi + 1] =
          data[i00 + 1]! * w00 + data[i10 + 1]! * w10 + data[i01 + 1]! * w01 + data[i11 + 1]! * w11;
        out.data[oi + 2] =
          data[i00 + 2]! * w00 + data[i10 + 2]! * w10 + data[i01 + 2]! * w01 + data[i11 + 2]! * w11;
      }
    }
  } else if (c === 1) {
    // Fast path: inline bilinear for 1-channel
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        const fx = mapX[idx]!;
        const fy = mapY[idx]!;
        const x0 = Math.floor(fx),
          y0 = Math.floor(fy);
        const wx = fx - x0,
          wy = fy - y0;

        const rx0 = reflectCoord(x0, w),
          rx1 = reflectCoord(x0 + 1, w);
        const ry0 = reflectCoord(y0, h),
          ry1 = reflectCoord(y0 + 1, h);

        out.data[idx] =
          data[ry0 * w + rx0]! * (1 - wx) * (1 - wy) +
          data[ry0 * w + rx1]! * wx * (1 - wy) +
          data[ry1 * w + rx0]! * (1 - wx) * wy +
          data[ry1 * w + rx1]! * wx * wy;
      }
    }
  } else {
    // Generic fallback
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        const px = sampleBilinear(img, mapX[idx]!, mapY[idx]!);
        const oi = idx * c;
        for (let ch = 0; ch < c; ch++) out.data[oi + ch] = px[ch]!;
      }
    }
  }
  return out;
}

export async function remap(img: Img, mapX: Float32Array, mapY: Float32Array): Promise<Img> {
  const gpu = await gpuRemap(img, mapX, mapY);
  if (gpu) return gpu;
  return remapCpu(img, mapX, mapY);
}

function convolve2dCpu(img: Img, kernel: ArrayLike<number>, kw: number, kh: number): Img {
  const out = createImg(img.w, img.h, img.c);
  const { w, h, c } = img;
  const khh = (kh - 1) >> 1,
    kwh = (kw - 1) >> 1;

  // Precompute sparse kernel: only non-zero entries (huge win for motion blur
  // where a 25x25 grid has ~25 non-zero entries out of 625)
  const sparseKy: number[] = [];
  const sparseKx: number[] = [];
  const sparseW: number[] = [];
  for (let ky = 0; ky < kh; ky++) {
    for (let kx = 0; kx < kw; kx++) {
      const v = kernel[ky * kw + kx]!;
      if (v !== 0) {
        sparseKy.push(ky - khh);
        sparseKx.push(kx - kwh);
        sparseW.push(v);
      }
    }
  }
  const sparseLen = sparseW.length;

  if (c === 1) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let i = 0; i < sparseLen; i++) {
          const sy = reflectCoord(y + sparseKy[i]!, h);
          const sx = reflectCoord(x + sparseKx[i]!, w);
          sum += img.data[sy * w + sx]! * sparseW[i]!;
        }
        out.data[y * w + x] = sum;
      }
    }
  } else if (c === 3) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s0 = 0,
          s1 = 0,
          s2 = 0;
        for (let i = 0; i < sparseLen; i++) {
          const sy = reflectCoord(y + sparseKy[i]!, h);
          const sx = reflectCoord(x + sparseKx[i]!, w);
          const idx = (sy * w + sx) * 3;
          const wt = sparseW[i]!;
          s0 += img.data[idx]! * wt;
          s1 += img.data[idx + 1]! * wt;
          s2 += img.data[idx + 2]! * wt;
        }
        const oi = (y * w + x) * 3;
        out.data[oi] = s0;
        out.data[oi + 1] = s1;
        out.data[oi + 2] = s2;
      }
    }
  } else {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        for (let ch = 0; ch < c; ch++) {
          let sum = 0;
          for (let i = 0; i < sparseLen; i++) {
            const sy = reflectCoord(y + sparseKy[i]!, h);
            const sx = reflectCoord(x + sparseKx[i]!, w);
            sum += img.data[(sy * w + sx) * c + ch]! * sparseW[i]!;
          }
          out.data[(y * w + x) * c + ch] = sum;
        }
      }
    }
  }
  return out;
}

export async function convolve2d(
  img: Img,
  kernel: ArrayLike<number>,
  kw: number,
  kh: number,
): Promise<Img> {
  const gpu = await gpuConvolve2d(img, kernel, kw, kh);
  if (gpu) return gpu;
  return convolve2dCpu(img, kernel, kw, kh);
}

export function generateClouds(h: number, w: number, seed = 0): Img {
  const rng = new SeededRNG(seed);
  const result = new Float32Array(w * h);

  for (let o = 0; o < 7; o++) {
    const f = 2 ** o;
    const sh = Math.max(2, Math.floor(h / (128 / f)));
    const sw = Math.max(2, Math.floor(w / (128 / f)));
    const small = new Float32Array(sh * sw);
    for (let i = 0; i < small.length; i++) small[i] = rng.randn();

    const weight = 1 / (f * 0.7 + 1);
    for (let y = 0; y < h; y++) {
      const fy = (y * (sh - 1)) / (h - 1 || 1);
      const y0 = Math.floor(fy),
        y1 = Math.min(y0 + 1, sh - 1);
      const wy = fy - y0;
      for (let x = 0; x < w; x++) {
        const fx = (x * (sw - 1)) / (w - 1 || 1);
        const x0 = Math.floor(fx),
          x1 = Math.min(x0 + 1, sw - 1);
        const wx = fx - x0;
        const v =
          small[y0 * sw + x0]! * (1 - wx) * (1 - wy) +
          small[y0 * sw + x1]! * wx * (1 - wy) +
          small[y1 * sw + x0]! * (1 - wx) * wy +
          small[y1 * sw + x1]! * wx * wy;
        result[y * w + x] += v * weight;
      }
    }
  }

  let mn = Infinity,
    mx = -Infinity;
  for (let i = 0; i < result.length; i++) {
    if (result[i]! < mn) mn = result[i]!;
    if (result[i]! > mx) mx = result[i]!;
  }
  const out = createImg(w, h, 1);
  const range = mx - mn || 1;
  for (let i = 0; i < result.length; i++) {
    out.data[i] = ((result[i]! - mn) / range) * 255;
  }
  return out;
}

export async function ripple(
  img: Img,
  amount: number,
  size: "small" | "medium" | "large" = "large",
): Promise<Img> {
  const wl = { small: 8, medium: 18, large: 40 }[size];
  const { w, h } = img;
  const rng = new SeededRNG(123);

  const yJitter = new Float32Array(h);
  const xJitter = new Float32Array(w);
  for (let i = 0; i < h; i++) yJitter[i] = rng.uniform(0.85, 1.15);
  for (let i = 0; i < w; i++) xJitter[i] = rng.uniform(0.85, 1.15);

  // Precompute per-row dx and per-column dy
  const rowDx = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    rowDx[y] = amount * Math.sin((2 * Math.PI * y) / (wl * yJitter[y]!));
  }
  const colDy = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    colDy[x] = amount * Math.sin((2 * Math.PI * x) / (wl * xJitter[x]!));
  }

  // Try GPU ripple (uploads only h+w floats instead of 2*w*h)
  const gpu = await gpuRipple(img, rowDx, colDy);
  if (gpu) return gpu;

  // CPU fallback: build full maps
  const mapX = new Float32Array(w * h);
  const mapY = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const dx = rowDx[y]!;
    const rowOff = y * w;
    for (let x = 0; x < w; x++) {
      mapX[rowOff + x] = x + dx;
      mapY[rowOff + x] = y + colDy[x]!;
    }
  }
  return remapCpu(img, mapX, mapY);
}

export function buildMotionKernel(
  angleDeg: number,
  distance: number,
): { kernel: Float64Array; ks: number } {
  const dist = Math.max(1, Math.round(distance));
  const ks = dist * 2 + 1;
  const kernel = new Float64Array(ks * ks);
  const rad = (angleDeg * Math.PI) / 180;
  const c = Math.cos(rad),
    s = Math.sin(rad);
  const ctr = dist;
  const numSamples = ks * 4;
  for (let i = 0; i < numSamples; i++) {
    const t = (i / (numSamples - 1)) * (ks - 1) - ctr;
    const fx = ctr + t * c;
    const fy = ctr - t * s;
    const x0 = Math.floor(fx),
      y0 = Math.floor(fy);
    const wx = fx - x0,
      wy = fy - y0;
    for (const [yy, yw] of [
      [y0, 1 - wy],
      [y0 + 1, wy],
    ] as [number, number][]) {
      for (const [xx, xw] of [
        [x0, 1 - wx],
        [x0 + 1, wx],
      ] as [number, number][]) {
        if (xx >= 0 && xx < ks && yy >= 0 && yy < ks) {
          kernel[yy * ks + xx] += yw * xw;
        }
      }
    }
  }
  let sum = 0;
  for (let i = 0; i < kernel.length; i++) sum += kernel[i]!;
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  return { kernel, ks };
}

export async function motionBlur(img: Img, angleDeg: number, distance: number): Promise<Img> {
  const { kernel, ks } = buildMotionKernel(angleDeg, distance);
  return convolve2d(img, kernel, ks, ks);
}

export async function displace(img: Img, dmap: Img, hScale: number, vScale: number): Promise<Img> {
  const { w, h } = img;
  const mapX = new Float32Array(w * h);
  const mapY = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const hv = dmap.data[idx * dmap.c]!;
      const vv = dmap.data[idx * dmap.c + 1]!;
      mapX[idx] = x + ((hv - 128) / 128) * hScale;
      mapY[idx] = y + ((vv - 128) / 128) * vScale;
    }
  }
  return remap(img, mapX, mapY);
}

export function mezzotint(gray: Img, rng: SeededRNG): Img {
  const out = createImg(gray.w, gray.h, 1);
  for (let i = 0; i < gray.data.length; i++) {
    out.data[i] = rng.randint(256) < gray.data[i]! ? 255 : 0;
  }
  return out;
}

export async function sobel5x5(gray: Img): Promise<Img> {
  const kx = [
    -1, -2, 0, 2, 1, -4, -8, 0, 8, 4, -6, -12, 0, 12, 6, -4, -8, 0, 8, 4, -1, -2, 0, 2, 1,
  ];
  const ky = [
    -1, -4, -6, -4, -1, -2, -8, -12, -8, -2, 0, 0, 0, 0, 0, 2, 8, 12, 8, 2, 1, 4, 6, 4, 1,
  ];
  const gx = await convolve2d(gray, kx, 5, 5);
  const gy = await convolve2d(gray, ky, 5, 5);
  const out = createImg(gray.w, gray.h, 1);
  for (let i = 0; i < out.data.length; i++) {
    out.data[i] = Math.sqrt(gx.data[i]! ** 2 + gy.data[i]! ** 2);
  }
  let mx = 0;
  for (let i = 0; i < out.data.length; i++) if (out.data[i]! > mx) mx = out.data[i]!;
  if (mx > 0) for (let i = 0; i < out.data.length; i++) out.data[i] = (out.data[i]! / mx) * 255;
  return out;
}

export function makeEllipseKernel(size: number): Uint8Array {
  const r = (size - 1) / 2;
  const mask = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (x - r) / r,
        dy = (y - r) / r;
      mask[y * size + x] = dx * dx + dy * dy <= 1 ? 1 : 0;
    }
  }
  return mask;
}

export async function erode(gray: Img, kernelSize: number): Promise<Img> {
  const mask = makeEllipseKernel(kernelSize);
  const gpu = await gpuErode(gray, kernelSize, mask);
  if (gpu) return gpu;

  const out = createImg(gray.w, gray.h, 1);
  const { w, h } = gray;
  const r = (kernelSize - 1) >> 1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let minVal = 255;
      for (let ky = 0; ky < kernelSize; ky++) {
        for (let kx = 0; kx < kernelSize; kx++) {
          if (!mask[ky * kernelSize + kx]) continue;
          const sy = reflectCoord(y + ky - r, h);
          const sx = reflectCoord(x + kx - r, w);
          const v = gray.data[sy * w + sx]!;
          if (v < minVal) minVal = v;
        }
      }
      out.data[y * w + x] = minVal;
    }
  }
  return out;
}

/**
 * Dilate a 1-channel image with an elliptical structuring element.
 * For large kernels (>15), uses a two-pass separable rectangular max
 * approximation for speed, then masks to the ellipse on a final pass.
 */
export async function dilate(gray: Img, kernelSize: number): Promise<Img> {
  const mask = makeEllipseKernel(kernelSize);

  // Try GPU first
  const gpu = await gpuDilate(gray, kernelSize, mask);
  if (gpu) return gpu;

  const { w, h } = gray;

  if (kernelSize <= 15) {
    // Small kernel: brute-force (original approach)
    const out = createImg(w, h, 1);
    const r = (kernelSize - 1) >> 1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let maxVal = 0;
        for (let ky = 0; ky < kernelSize; ky++) {
          for (let kx = 0; kx < kernelSize; kx++) {
            if (!mask[ky * kernelSize + kx]) continue;
            const sy = reflectCoord(y + ky - r, h);
            const sx = reflectCoord(x + kx - r, w);
            const v = gray.data[sy * w + sx]!;
            if (v > maxVal) maxVal = v;
          }
        }
        out.data[y * w + x] = maxVal;
      }
    }
    return out;
  }

  // Large kernel: decompose ellipse into per-row horizontal spans,
  // then for each row offset, do a 1D sliding-max (Van Herk) across
  // the corresponding horizontal span width. This gives exact elliptical
  // dilation in O(h * kernelSize + w * h) instead of O(w * h * ks^2).
  const r = (kernelSize - 1) >> 1;

  // Precompute per-row x-span of the ellipse
  const rowMinX = new Int32Array(kernelSize);
  const rowMaxX = new Int32Array(kernelSize);
  for (let ky = 0; ky < kernelSize; ky++) {
    let minX = kernelSize,
      maxX = -1;
    for (let kx = 0; kx < kernelSize; kx++) {
      if (mask[ky * kernelSize + kx]) {
        if (kx < minX) minX = kx;
        if (kx > maxX) maxX = kx;
      }
    }
    rowMinX[ky] = minX - r;
    rowMaxX[ky] = maxX - r;
  }

  // For each kernel row offset, compute horizontal 1D max with the
  // corresponding span width using a deque-based sliding window.
  const out = createImg(w, h, 1);
  const temp = new Float32Array(w);
  const deque = new Int32Array(w); // indices

  for (let ky = 0; ky < kernelSize; ky++) {
    const spanLeft = rowMinX[ky]!;
    const spanRight = rowMaxX[ky]!;
    if (spanRight < spanLeft) continue; // empty row in ellipse
    const dy = ky - r;

    for (let y = 0; y < h; y++) {
      const sy = reflectCoord(y + dy, h);
      const srcRow = sy * w;

      // Sliding max over gray row `sy` with window [x+spanLeft, x+spanRight]
      let dqFront = 0,
        dqBack = 0;
      for (let x = 0; x < w; x++) {
        const winRight = x + spanRight;
        const winLeft = x + spanLeft;

        // Add new element at winRight
        if (winRight < w) {
          const val = gray.data[srcRow + winRight]!;
          while (dqBack > dqFront && gray.data[srcRow + deque[dqBack - 1]!]! <= val) dqBack--;
          deque[dqBack++] = winRight;
        } else {
          const rx = reflectCoord(winRight, w);
          // For reflected coords, fall back to direct comparison
          const val = gray.data[srcRow + rx]!;
          while (dqBack > dqFront && gray.data[srcRow + deque[dqBack - 1]!]! <= val) dqBack--;
          deque[dqBack++] = winRight; // store original index for eviction
        }

        // Remove elements outside window
        while (dqFront < dqBack && deque[dqFront]! < winLeft) dqFront++;

        temp[x] = gray.data[srcRow + reflectCoord(deque[dqFront]!, w)]!;
      }

      // Max with existing output
      const outRow = y * w;
      for (let x = 0; x < w; x++) {
        if (temp[x]! > out.data[outRow + x]!) {
          out.data[outRow + x] = temp[x]!;
        }
      }
    }
  }

  return out;
}
