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
} from "./effects.ts";
import { loadImageFromUrl, gaussianBlur, resizeImg } from "./canvas-io.ts";
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

function yieldToUI(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

// Cache the raw spray map so we don't re-fetch on every run
let sprayMapCache: Img | null = null;

async function getSprayMap(): Promise<Img> {
  if (sprayMapCache) return sprayMapCache;
  sprayMapCache = await loadImageFromUrl("/SprayMap_composite.png");
  return sprayMapCache;
}

export async function sprayAnything(
  inputImg: Img,
  params: SprayParams,
  options: SprayOptions = {},
): Promise<SprayResult> {
  const { onProgress = () => {}, onDebugStep } = options;
  const timings: StepTiming[] = [];
  const totalStart = performance.now();

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
  await yieldToUI();

  const sprayMapFull = await getSprayMap();
  const sprayMap = resizeImg(sprayMapFull, w, h);
  const original = cloneImg(img);

  await dbg("original", img);
  await dbg("spray map", sprayMap);

  // 1. Initial distortion
  onProgress("Ripple distortion...", 5);
  await yieldToUI();
  img = await ripple(img, params.rippleAmount, params.rippleSize);
  await dbg("ripple", img);

  onProgress("Motion blur...", 10);
  await yieldToUI();
  img = roll(img, params.rollDx, params.rollDy);
  img = await motionBlur(img, params.motionBlurAngle, params.motionBlurDistance);
  await dbg("offset + motion blur", img);

  // 2. Displacement passes
  for (let i = 0; i < params.dispPasses.length; i++) {
    const [hs, vs, fp] = params.dispPasses[i]!;
    onProgress(`Displacement pass ${i + 1}/${params.dispPasses.length}...`, 15 + i * 5);
    await yieldToUI();
    const pre = cloneImg(img);
    img = await displace(img, sprayMap, hs, vs);
    img = fade(pre, img, fp);
    await dbg(`displace (${hs},${vs}) fade ${fp}%`, img);
  }

  await dbg("6-pass displaced", cloneImg(img));

  onProgress("Blending displaced...", 45);
  await yieldToUI();
  img = fade(original, img, params.displacedBlendPct);
  const displaced = cloneImg(img);
  await dbg("blend displaced", img);

  // 3. Spray coverage texture
  onProgress("Generating clouds...", 50);
  await yieldToUI();
  const clouds = generateClouds(h, w, params.cloudsSeed);
  await dbg("clouds", clouds);

  onProgress("Mezzotint...", 53);
  await yieldToUI();
  const rngMezz = new SeededRNG(params.mezzSeed);
  const mezz = mezzotint(clouds, rngMezz);
  await dbg("mezzotint", mezz);

  let coverage: Img = fade(clouds, mezz, params.coverageMezzBlend);
  coverage = levels(coverage, params.coverageLevelsLo, params.coverageLevelsHi);
  coverage = await motionBlur(coverage, params.coverageMotionAngle, params.coverageMotionDist);
  await dbg("coverage mask", coverage);

  onProgress("Applying coverage...", 56);
  await yieldToUI();
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
  await dbg("coverage applied", img);

  // 4. Edge speckles
  onProgress("Edge detection...", 60);
  await yieldToUI();
  const grayOrig = toGrayscale(original);
  const grad = await sobel5x5(grayOrig);

  onProgress("Dilating edges...", 65);
  await yieldToUI();
  const edgeZone = await dilate(grad, params.edgeDilateSize);
  const edgeBlurred = gaussianBlur(grayToRgb(edgeZone), params.edgeBlurSigma);
  const edgeF = toGrayscale(edgeBlurred);
  for (let i = 0; i < edgeF.data.length; i++) edgeF.data[i] /= 255;
  await dbg("edge zone", edgeZone);

  onProgress("Generating speckles...", 70);
  await yieldToUI();
  const rngDots = new SeededRNG(777);
  const fineDots = createImg(w, h, 1);
  for (let i = 0; i < n; i++) {
    fineDots.data[i] = clamp((rngDots.randint(256) * params.fineDotsContrast) / 100);
  }
  let dots1 = threshold(fineDots, params.fineDotsThreshold);
  dots1 = erode(dots1, params.fineDotsErodeSize);
  dots1 = await ripple(dots1, params.fineDotsRipple, "medium");
  await dbg("fine dots", dots1);

  onProgress("Large dots...", 75);
  await yieldToUI();
  function makeDots(rng: SeededRNG): Img {
    const d = createImg(w, h, 1);
    for (let i = 0; i < n; i++) {
      d.data[i] = clamp(128 + ((rng.randint(256) - 128) * params.largeDotsContrast) / 100);
    }
    return threshold(d, 253);
  }
  const rngD2a = new SeededRNG(params.largeDotsSeed1),
    rngD2b = new SeededRNG(params.largeDotsSeed2);
  let dots2 = bitwiseOr(makeDots(rngD2a), makeDots(rngD2b));
  dots2 = await dilate(dots2, params.largeDotsDilateSize);
  dots2 = await ripple(dots2, params.largeDotsRipple, "medium");
  await dbg("large dots", dots2);

  const allDots = maxImages(dots1, dots2);
  const speckleWeight = createImg(w, h, 1);
  for (let i = 0; i < n; i++) {
    speckleWeight.data[i] = Math.min(1, (allDots.data[i]! / 255) * edgeF.data[i]! * 2);
  }

  onProgress("Applying speckles...", 80);
  await yieldToUI();
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

  // 5. Paint grain
  onProgress("Paint grain...", 85);
  await yieldToUI();
  const gray = toGrayscale(img);
  const rngGrain = new SeededRNG(params.grainSeed);
  for (let i = 0; i < n; i++) {
    const g = rngGrain.randint(256);
    const mask = g < gray.data[i]! ? 1 : 0;
    for (let ch = 0; ch < 3; ch++) {
      const v = img.data[i * 3 + ch]!;
      img.data[i * 3 + ch] = v * mask + v * params.grainDarkFactor * (1 - mask);
    }
  }
  await dbg("paint grain", img);

  // 6. Sharpen
  onProgress("Sharpening...", 92);
  await yieldToUI();
  const blurred04 = gaussianBlur(img, params.sharpenSigma1);
  const blurred2 = gaussianBlur(blurred04, params.sharpenSigma2);
  for (let i = 0; i < blurred04.data.length; i++) {
    img.data[i] = clamp(blurred04.data[i]! + (blurred04.data[i]! - blurred2.data[i]!));
  }
  await dbg("sharpen (final)", img);

  onProgress("Done!", 100);
  const totalMs = performance.now() - totalStart;
  return { img, timings, totalMs };
}
