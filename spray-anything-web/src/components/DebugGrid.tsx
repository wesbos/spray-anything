import { useEffect, useRef, useState } from "react";
import type { Img } from "../image.ts";
import { grayToRgb, cloneImg } from "../image.ts";
import { imgToBlob } from "../canvas-io.ts";
import type { StepTiming } from "../pipeline.ts";

export interface DebugStep {
  label: string;
  img: Img;
  timing: StepTiming;
}

function DebugStepCard({ step, index }: { step: DebugStep; index: number }) {
  const [url, setUrl] = useState<string | null>(null);
  const urlRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const vis = step.img.c === 1 ? grayToRgb(step.img) : cloneImg(step.img);
    void imgToBlob(vis).then((blob) => {
      if (cancelled) return;
      const u = URL.createObjectURL(blob);
      urlRef.current = u;
      setUrl(u);
    });
    return () => {
      cancelled = true;
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    };
  }, [step]);

  return (
    <div className="debug-step">
      {url ? <img src={url} alt={step.label} /> : <div className="debug-step-placeholder" />}
      <span>
        {index + 1}. {step.label}
      </span>
      <span className="debug-timing">{step.timing.durationMs.toFixed(0)}ms</span>
    </div>
  );
}

interface DebugGridProps {
  steps: DebugStep[];
}

export function DebugGrid({ steps }: DebugGridProps) {
  if (steps.length === 0) return null;

  return (
    <div className="debug-area">
      <h2>Debug Steps</h2>
      <div className="debug-grid">
        {steps.map((step, i) => (
          <DebugStepCard key={`${i}-${step.label}`} step={step} index={i} />
        ))}
      </div>
    </div>
  );
}
