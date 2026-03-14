export interface SprayParams {
  // 1. Initial distortion
  rippleAmount: number;
  rippleSize: "small" | "medium" | "large";
  rollDx: number;
  rollDy: number;
  motionBlurAngle: number;
  motionBlurDistance: number;

  // 2. Displacement
  dispPasses: [number, number, number][]; // [hScale, vScale, fadePct]
  displacedBlendPct: number;

  // 3. Coverage
  cloudsSeed: number;
  mezzSeed: number;
  coverageMezzBlend: number;
  coverageLevelsLo: number;
  coverageLevelsHi: number;
  coverageMotionAngle: number;
  coverageMotionDist: number;
  slightOriginalPct: number;

  // 4. Edge speckles
  edgeDilateSize: number;
  edgeBlurSigma: number;
  fineDotsContrast: number;
  fineDotsThreshold: number;
  fineDotsErodeSize: number;
  fineDotsRipple: number;
  largeDotsSeed1: number;
  largeDotsSeed2: number;
  largeDotsContrast: number;
  largeDotsDilateSize: number;
  largeDotsRipple: number;
  speckleDispScale: number;

  // 5. Paint grain
  grainSeed: number;
  grainDarkFactor: number;

  // 6. Sharpen
  sharpenSigma1: number;
  sharpenSigma2: number;

  // Input
  maxDim: number;
}

export const DEFAULT_PARAMS: SprayParams = {
  rippleAmount: 14,
  rippleSize: "large",
  rollDx: 6,
  rollDy: 6,
  motionBlurAngle: -27,
  motionBlurDistance: 12,

  dispPasses: [
    [120, 120, 75],
    [120, -120, 75],
    [0, 120, 75],
    [120, 0, 75],
    [999, 999, 75],
    [-999, 999, 25],
  ],
  displacedBlendPct: 70,

  cloudsSeed: 42,
  mezzSeed: 99,
  coverageMezzBlend: 50,
  coverageLevelsLo: 8,
  coverageLevelsHi: 194,
  coverageMotionAngle: -27,
  coverageMotionDist: 6,
  slightOriginalPct: 90,

  edgeDilateSize: 51,
  edgeBlurSigma: 20,
  fineDotsContrast: 52,
  fineDotsThreshold: 128,
  fineDotsErodeSize: 3,
  fineDotsRipple: 150,
  largeDotsSeed1: 888,
  largeDotsSeed2: 999,
  largeDotsContrast: 50,
  largeDotsDilateSize: 13,
  largeDotsRipple: 150,
  speckleDispScale: 150,

  grainSeed: 555,
  grainDarkFactor: 0.85,

  sharpenSigma1: 0.4,
  sharpenSigma2: 2.0,

  maxDim: 1500,
};
