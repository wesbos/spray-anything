import {
  type Img,
  createImg,
  cloneImg,
  clamp,
  toGrayscale,
  grayToRgb,
  fade,
  roll,
  threshold,
  maxImages,
  bitwiseOr,
  levels,
} from "./image.ts";
import { SeededRNG } from "./rng.ts";
import {
  ripple,
  motionBlur,
  displace,
  mezzotint,
  sobel5x5,
  dilate,
  erode,
  generateClouds,
  buildMotionKernel,
} from "./effects.ts";
import { loadImageFromUrl, gaussianBlur, resizeImg } from "./canvas-io.ts";
import {
  gpuPaintGrain,
  gpuSharpen,
  gpuFusedDistortion,
  flushBufferPool,
  warmUpGpu,
} from "./webgpu.ts";
import type { SprayParams } from "./params.ts";

export interface StepTiming {
  label: string;
  durationMs: number;
}

export type ProgressCallback = (step: string, pct: number) => void;
export type DebugCallback = (label: string, img: Img, timing: StepTiming) => Promise<void> | void;

export interface SprayOptions {
  onProgress?: ProgressCallback;
  onDebugStep?: DebugCallback;
}

export interface SprayResult {
  img: Img;
  timings: StepTiming[];
  totalMs: number;
}

// Cache the raw spray map and the last resized version
let sprayMapRaw: Img | null = null;
let sprayMapResized: Img | null = null;
let sprayMapResizedKey = "";

export async function preloadSprayMap(): Promise<void> {
  if (!sprayMapRaw) sprayMapRaw = await loadImageFromUrl("/SprayMap_composite.png");
}

async function getResizedSprayMap(w: number, h: number): Promise<Img> {
  const key = `${w}x${h}`;
  if (sprayMapResized && sprayMapResizedKey === key) return sprayMapResized;
  if (!sprayMapRaw) sprayMapRaw = await loadImageFromUrl("/SprayMap_composite.png");
  sprayMapResized = resizeImg(sprayMapRaw, w, h);
  sprayMapResizedKey = key;
  return sprayMapResized;
}

export async function sprayAnything(
  inputImg: Img,
  params: SprayParams,
  options: SprayOptions = {},
): Promise<SprayResult> {
  const { onProgress = () => {}, onDebugStep } = options;
  const timings: StepTiming[] = [];
  const totalStart = performance.now();
  const isDebug = !!onDebugStep;

  let stepStart = performance.now();
  const dbg = async (label: string, img: Img) => {
    const durationMs = performance.now() - stepStart;
    const timing: StepTiming = { label, durationMs };
    timings.push(timing);
    if (onDebugStep) await onDebugStep(label, img, timing);
    stepStart = performance.now();
  };

  // Cap input size
  let img: Img;
  if (inputImg.w > params.maxDim || inputImg.h > params.maxDim) {
    const scale = params.maxDim / Math.max(inputImg.w, inputImg.h);
    const nw = Math.round(inputImg.w * scale);
    const nh = Math.round(inputImg.h * scale);
    img = resizeImg(inputImg, nw, nh);
  } else {
    img = cloneImg(inputImg);
  }

  const { w, h } = img;
  const n = w * h;
  onProgress("Loading spray map...", 0);

  // Ensure GPU pipelines are compiled before first dispatch
  await warmUpGpu();

  const sprayMap = await getResizedSprayMap(w, h);
  const original = cloneImg(img);

  await dbg("original", img);

  // ── Start edge/dots branch early (only needs `original`) ───────────────
  // This runs in parallel with distortion + displacement + coverage
  const edgeBranchPromise = (async () => {
    const grayOrig = toGrayscale(original);
    const [edgeF, dots1Result, dots2Result] = await Promise.all([
      (async () => {
        const grad = await sobel5x5(grayOrig);
        const edgeZone = await dilate(grad, params.edgeDilateSize);
        const edgeBlurred = await gaussianBlur(grayToRgb(edgeZone), params.edgeBlurSigma);
        const ef = toGrayscale(edgeBlurred);
        for (let i = 0; i < ef.data.length; i++) ef.data[i] /= 255;
        if (isDebug) await dbg("edge zone", edgeZone);
        return ef;
      })(),
      (async () => {
        const rngDots = new SeededRNG(777);
        const fineDots = createImg(w, h, 1);
        for (let i = 0; i < n; i++) {
          fineDots.data[i] = clamp((rngDots.randint(256) * params.fineDotsContrast) / 100);
        }
        let d1 = threshold(fineDots, params.fineDotsThreshold);
        d1 = await erode(d1, params.fineDotsErodeSize);
        d1 = await ripple(d1, params.fineDotsRipple, "medium");
        if (isDebug) await dbg("fine dots", d1);
        return d1;
      })(),
      (async () => {
        function makeDots(rng: SeededRNG): Img {
          const d = createImg(w, h, 1);
          for (let i = 0; i < n; i++) {
            d.data[i] = clamp(128 + ((rng.randint(256) - 128) * params.largeDotsContrast) / 100);
          }
          return threshold(d, 253);
        }
        const rngD2a = new SeededRNG(params.largeDotsSeed1);
        const rngD2b = new SeededRNG(params.largeDotsSeed2);
        let d2 = bitwiseOr(makeDots(rngD2a), makeDots(rngD2b));
        d2 = await dilate(d2, params.largeDotsDilateSize);
        d2 = await ripple(d2, params.largeDotsRipple, "medium");
        if (isDebug) await dbg("large dots", d2);
        return d2;
      })(),
    ]);
    return { edgeF, dots1: dots1Result, dots2: dots2Result };
  })();

  // ── Main pipeline (runs in parallel with edge branch) ──────────────────

  // 1+2. Fused: ripple → roll → motionBlur → 6x displace+fade
  //       Single GPU upload, all passes on GPU, single readback
  onProgress("Distortion + displacement...", 5);
  {
    const wl = { small: 8, medium: 18, large: 40 }[params.rippleSize];
    const rngRipple = new SeededRNG(123);
    const yJitter = new Float32Array(h);
    const xJitter = new Float32Array(w);
    for (let i = 0; i < h; i++) yJitter[i] = rngRipple.uniform(0.85, 1.15);
    for (let i = 0; i < w; i++) xJitter[i] = rngRipple.uniform(0.85, 1.15);
    const rowDx = new Float32Array(h);
    for (let y = 0; y < h; y++) {
      rowDx[y] = params.rippleAmount * Math.sin((2 * Math.PI * y) / (wl * yJitter[y]!));
    }
    const colDy = new Float32Array(w);
    for (let x = 0; x < w; x++) {
      colDy[x] = params.rippleAmount * Math.sin((2 * Math.PI * x) / (wl * xJitter[x]!));
    }

    const { kernel: motionKernel, ks } = buildMotionKernel(
      params.motionBlurAngle,
      params.motionBlurDistance,
    );
    const khh = (ks - 1) >> 1;
    const kwh = (ks - 1) >> 1;
    const tapsList: number[] = [];
    for (let ky = 0; ky < ks; ky++)
      for (let kx = 0; kx < ks; kx++) {
        const v = motionKernel[ky * ks + kx]!;
        if (v !== 0) tapsList.push(ky - khh, kx - kwh, v);
      }

    const fusedResult = await gpuFusedDistortion(
      img,
      sprayMap,
      rowDx,
      colDy,
      params.rollDx,
      params.rollDy,
      new Float32Array(tapsList),
      tapsList.length / 3,
      params.dispPasses,
    );
    if (fusedResult) {
      img = fusedResult;
    } else {
      // CPU fallback
      img = await ripple(img, params.rippleAmount, params.rippleSize);
      img = roll(img, params.rollDx, params.rollDy);
      img = await motionBlur(img, params.motionBlurAngle, params.motionBlurDistance);
      for (let i = 0; i < params.dispPasses.length; i++) {
        const [hs, vs, fp] = params.dispPasses[i]!;
        const pre = cloneImg(img);
        img = await displace(img, sprayMap, hs, vs);
        img = fade(pre, img, fp);
      }
    }
  }
  await dbg("distortion + displacement", img);

  onProgress("Blending displaced...", 45);
  img = fade(original, img, params.displacedBlendPct);
  const displaced = cloneImg(img);
  await dbg("blend displaced", img);

  // 3. Coverage (needs displaced, runs while we wait for edge branch)
  onProgress("Coverage...", 50);
  const clouds = generateClouds(h, w, params.cloudsSeed);
  if (isDebug) await dbg("clouds", clouds);
  const rngMezz = new SeededRNG(params.mezzSeed);
  const mezz = mezzotint(clouds, rngMezz);
  if (isDebug) await dbg("mezzotint", mezz);
  let coverage: Img = fade(clouds, mezz, params.coverageMezzBlend);
  coverage = levels(coverage, params.coverageLevelsLo, params.coverageLevelsHi);
  coverage = await motionBlur(coverage, params.coverageMotionAngle, params.coverageMotionDist);
  if (isDebug) await dbg("coverage mask", coverage);
  const slightOriginal = fade(original, displaced, params.slightOriginalPct);
  const covOut = createImg(w, h, 3);
  for (let i = 0; i < n; i++) {
    const cf = coverage.data[i]! / 255;
    for (let ch = 0; ch < 3; ch++) {
      covOut.data[i * 3 + ch] =
        displaced.data[i * 3 + ch]! * cf + slightOriginal.data[i * 3 + ch]! * (1 - cf);
    }
  }
  img = covOut;
  if (isDebug) await dbg("coverage applied", img);

  // ── Wait for edge branch to finish ─────────────────────────────────────
  const edgeStart = performance.now();
  const { edgeF, dots1, dots2 } = await edgeBranchPromise;
  timings.push({ label: "edge/dots wait", durationMs: performance.now() - edgeStart });
  stepStart = performance.now();

  // Combine speckle results
  onProgress("Applying speckles...", 80);
  const allDots = maxImages(dots1, dots2);
  const speckleWeight = createImg(w, h, 1);
  for (let i = 0; i < n; i++) {
    speckleWeight.data[i] = Math.min(1, (allDots.data[i]! / 255) * edgeF.data[i]! * 2);
  }
  const heavyDisp = await displace(
    displaced,
    sprayMap,
    params.speckleDispScale,
    params.speckleDispScale,
  );
  for (let i = 0; i < n; i++) {
    const sw = speckleWeight.data[i]!;
    for (let ch = 0; ch < 3; ch++) {
      img.data[i * 3 + ch] = heavyDisp.data[i * 3 + ch]! * sw + img.data[i * 3 + ch]! * (1 - sw);
    }
  }
  await dbg("speckles applied", img);

  // 5. Paint grain (GPU fused)
  onProgress("Paint grain...", 85);
  const rngGrain = new SeededRNG(params.grainSeed);
  const rngValues = new Float32Array(n);
  for (let i = 0; i < n; i++) rngValues[i] = rngGrain.randint(256);
  const grainResult = await gpuPaintGrain(img, rngValues, params.grainDarkFactor);
  if (grainResult) {
    img = grainResult;
  } else {
    const gray = toGrayscale(img);
    for (let i = 0; i < n; i++) {
      const mask = rngValues[i]! < gray.data[i]! ? 1 : 0;
      for (let ch = 0; ch < 3; ch++) {
        const v = img.data[i * 3 + ch]!;
        img.data[i * 3 + ch] = v * mask + v * params.grainDarkFactor * (1 - mask);
      }
    }
  }
  await dbg("paint grain", img);

  // 6. Sharpen (GPU fused: 4 blur passes + unsharp mask)
  onProgress("Sharpening...", 92);
  const sharpenResult = await gpuSharpen(img, params.sharpenSigma1, params.sharpenSigma2);
  if (sharpenResult) {
    img = sharpenResult;
  } else {
    const blurred04 = await gaussianBlur(img, params.sharpenSigma1);
    const blurred2 = await gaussianBlur(blurred04, params.sharpenSigma2);
    for (let i = 0; i < blurred04.data.length; i++) {
      img.data[i] = clamp(blurred04.data[i]! + (blurred04.data[i]! - blurred2.data[i]!));
    }
  }
  await dbg("sharpen (final)", img);

  onProgress("Done!", 100);
  const totalMs = performance.now() - totalStart;
  flushBufferPool();
  return { img, timings, totalMs };
}
