import { useState } from "react";
import type { SprayParams } from "../params.ts";
import { DEFAULT_PARAMS } from "../params.ts";

interface ParameterEditorProps {
  params: SprayParams;
  onChange: (params: SprayParams) => void;
  disabled?: boolean;
}

interface SliderRowProps {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}

function SliderRow({ label, hint, value, min, max, step = 1, onChange, disabled }: SliderRowProps) {
  return (
    <div className="param-row-wrap">
      <label className="param-row">
        <span className="param-label">{label}</span>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        <span className="param-value">{value}</span>
      </label>
      {hint && <span className="param-hint">{hint}</span>}
    </div>
  );
}

interface SelectRowProps {
  label: string;
  hint?: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
  disabled?: boolean;
}

function SelectRow({ label, hint, value, options, onChange, disabled }: SelectRowProps) {
  return (
    <div className="param-row-wrap">
      <label className="param-row">
        <span className="param-label">{label}</span>
        <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
          {options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
        <span className="param-value" />
      </label>
      {hint && <span className="param-hint">{hint}</span>}
    </div>
  );
}

function Section({
  title,
  defaultOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="param-section">
      <button className="param-section-header" onClick={() => setOpen(!open)}>
        <span>{open ? "\u25BE" : "\u25B8"}</span> {title}
      </button>
      {open && <div className="param-section-body">{children}</div>}
    </div>
  );
}

export function ParameterEditor({ params, onChange, disabled }: ParameterEditorProps) {
  const set = <K extends keyof SprayParams>(key: K, value: SprayParams[K]) => {
    onChange({ ...params, [key]: value });
  };

  return (
    <div className="param-panels">
      <button
        className="btn-small btn-full"
        onClick={() => onChange({ ...DEFAULT_PARAMS })}
        disabled={disabled}
      >
        Reset to defaults
      </button>

      <Section title="Main Controls" defaultOpen>
        <SliderRow
          label="Distortion"
          hint="How wavy and warped the spray looks"
          value={params.rippleAmount}
          min={0}
          max={50}
          disabled={disabled}
          onChange={(v) => set("rippleAmount", v)}
        />
        <SliderRow
          label="Spray angle"
          hint="Direction of the spray strokes"
          value={params.motionBlurAngle}
          min={-180}
          max={180}
          disabled={disabled}
          onChange={(v) => {
            set("motionBlurAngle", v);
            // Also update coverage blur angle to match
            onChange({ ...params, motionBlurAngle: v, coverageMotionAngle: v });
          }}
        />
        <SliderRow
          label="Streak length"
          hint="How streaky/blurred the spray is"
          value={params.motionBlurDistance}
          min={0}
          max={30}
          disabled={disabled}
          onChange={(v) => set("motionBlurDistance", v)}
        />
        <SliderRow
          label="Effect strength"
          hint="Blend of spray effect vs original image"
          value={params.displacedBlendPct}
          min={0}
          max={100}
          disabled={disabled}
          onChange={(v) => set("displacedBlendPct", v)}
        />
        <SliderRow
          label="Coverage"
          hint="How much spray texture covers the image (low = more gaps)"
          value={params.coverageLevelsHi}
          min={128}
          max={255}
          disabled={disabled}
          onChange={(v) => set("coverageLevelsHi", v)}
        />
        <SliderRow
          label="Edge speckle spread"
          hint="How far scattered dots spread from edges"
          value={params.edgeDilateSize}
          min={3}
          max={101}
          step={2}
          disabled={disabled}
          onChange={(v) => set("edgeDilateSize", v)}
        />
        <SliderRow
          label="Paint grain"
          hint="Darkness of the grainy paint texture (1 = none)"
          value={params.grainDarkFactor}
          min={0.5}
          max={1}
          step={0.01}
          disabled={disabled}
          onChange={(v) => set("grainDarkFactor", v)}
        />
        <SliderRow
          label="Sharpness"
          hint="How sharp the final output is"
          value={params.sharpenSigma2}
          min={0.5}
          max={10}
          step={0.1}
          disabled={disabled}
          onChange={(v) => set("sharpenSigma2", v)}
        />
      </Section>

      <Section title="Advanced">
        <SelectRow
          label="Ripple size"
          value={params.rippleSize}
          options={["small", "medium", "large"]}
          disabled={disabled}
          onChange={(v) => set("rippleSize", v as SprayParams["rippleSize"])}
        />
        <SliderRow
          label="Roll X"
          value={params.rollDx}
          min={-20}
          max={20}
          disabled={disabled}
          onChange={(v) => set("rollDx", v)}
        />
        <SliderRow
          label="Roll Y"
          value={params.rollDy}
          min={-20}
          max={20}
          disabled={disabled}
          onChange={(v) => set("rollDy", v)}
        />
        <SliderRow
          label="Clouds seed"
          value={params.cloudsSeed}
          min={0}
          max={999}
          disabled={disabled}
          onChange={(v) => set("cloudsSeed", v)}
        />
        <SliderRow
          label="Mezz seed"
          value={params.mezzSeed}
          min={0}
          max={999}
          disabled={disabled}
          onChange={(v) => set("mezzSeed", v)}
        />
        <SliderRow
          label="Mezz blend %"
          value={params.coverageMezzBlend}
          min={0}
          max={100}
          disabled={disabled}
          onChange={(v) => set("coverageMezzBlend", v)}
        />
        <SliderRow
          label="Coverage low"
          value={params.coverageLevelsLo}
          min={0}
          max={128}
          disabled={disabled}
          onChange={(v) => set("coverageLevelsLo", v)}
        />
        <SliderRow
          label="Coverage blur dist"
          value={params.coverageMotionDist}
          min={0}
          max={20}
          disabled={disabled}
          onChange={(v) => set("coverageMotionDist", v)}
        />
        <SliderRow
          label="Original blend %"
          value={params.slightOriginalPct}
          min={0}
          max={100}
          disabled={disabled}
          onChange={(v) => set("slightOriginalPct", v)}
        />
        <SliderRow
          label="Edge blur"
          value={params.edgeBlurSigma}
          min={1}
          max={50}
          disabled={disabled}
          onChange={(v) => set("edgeBlurSigma", v)}
        />
        <SliderRow
          label="Fine dots contrast"
          value={params.fineDotsContrast}
          min={0}
          max={100}
          disabled={disabled}
          onChange={(v) => set("fineDotsContrast", v)}
        />
        <SliderRow
          label="Fine dots threshold"
          value={params.fineDotsThreshold}
          min={0}
          max={255}
          disabled={disabled}
          onChange={(v) => set("fineDotsThreshold", v)}
        />
        <SliderRow
          label="Fine dots ripple"
          value={params.fineDotsRipple}
          min={0}
          max={300}
          disabled={disabled}
          onChange={(v) => set("fineDotsRipple", v)}
        />
        <SliderRow
          label="Large dots dilate"
          value={params.largeDotsDilateSize}
          min={3}
          max={31}
          step={2}
          disabled={disabled}
          onChange={(v) => set("largeDotsDilateSize", v)}
        />
        <SliderRow
          label="Large dots ripple"
          value={params.largeDotsRipple}
          min={0}
          max={300}
          disabled={disabled}
          onChange={(v) => set("largeDotsRipple", v)}
        />
        <SliderRow
          label="Speckle scale"
          value={params.speckleDispScale}
          min={0}
          max={300}
          disabled={disabled}
          onChange={(v) => set("speckleDispScale", v)}
        />
        <SliderRow
          label="Grain seed"
          value={params.grainSeed}
          min={0}
          max={999}
          disabled={disabled}
          onChange={(v) => set("grainSeed", v)}
        />
        <SliderRow
          label="Sharpen sigma 1"
          value={params.sharpenSigma1}
          min={0.1}
          max={2}
          step={0.1}
          disabled={disabled}
          onChange={(v) => set("sharpenSigma1", v)}
        />
        <SliderRow
          label="Max dimension"
          value={params.maxDim}
          min={200}
          max={3000}
          step={100}
          disabled={disabled}
          onChange={(v) => set("maxDim", v)}
        />
      </Section>
    </div>
  );
}
