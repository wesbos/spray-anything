import { useState, useCallback, useRef, useEffect } from "react";
import type { Img } from "./image.ts";
import { cloneImg, grayToRgb } from "./image.ts";
import { loadImageFromFile, loadImageFromUrl, imgToBlob } from "./canvas-io.ts";
import { resizeImg } from "./canvas-io.ts";
import { preloadSprayMap } from "./pipeline.ts";
import { sprayAnything } from "./pipeline.ts";
import type { StepTiming } from "./pipeline.ts";
import { type SprayParams, DEFAULT_PARAMS } from "./params.ts";
import { saveGeneration } from "./history.ts";
import { DropZone } from "./components/DropZone.tsx";
import { ParameterEditor } from "./components/ParameterEditor.tsx";
import { DebugGrid, type DebugStep } from "./components/DebugGrid.tsx";
import { HistoryPanel } from "./components/HistoryPanel.tsx";

void preloadSprayMap();

type AppState = "idle" | "processing" | "done";

export function App() {
  const [state, setState] = useState<AppState>("idle");
  const [params, setParams] = useState<SprayParams>(DEFAULT_PARAMS);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [debug, setDebug] = useState(false);
  const [livePreview, setLivePreview] = useState(false);
  const [progressPct, setProgressPct] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [debugSteps, setDebugSteps] = useState<DebugStep[]>([]);
  const [timings, setTimings] = useState<StepTiming[]>([]);
  const [totalMs, setTotalMs] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [historyRefresh, setHistoryRefresh] = useState(0);
  const [isRegenerating, setIsRegenerating] = useState(false);

  const resultUrlRef = useRef<string | null>(null);
  const sourceImgRef = useRef<Img | null>(null);
  // Generation counter to discard stale results
  const genCounter = useRef(0);

  const cleanup = useCallback(() => {
    if (resultUrlRef.current) {
      URL.revokeObjectURL(resultUrlRef.current);
      resultUrlRef.current = null;
    }
  }, []);

  const processImage = useCallback(
    async (img: Img, currentParams: SprayParams, currentDebug: boolean) => {
      cleanup();
      setState("processing");
      setError(null);
      setProgressPct(0);
      setProgressLabel("Starting...");
      setDebugSteps([]);
      setTimings([]);
      setResultUrl(null);

      const gen = ++genCounter.current;

      try {
        const result = await sprayAnything(img, currentParams, {
          onProgress(step, pct) {
            if (genCounter.current !== gen) return;
            setProgressPct(pct);
            setProgressLabel(step);
          },
          onDebugStep: currentDebug
            ? async (label, stepImg, timing) => {
                if (genCounter.current !== gen) return;
                const vis = stepImg.c === 1 ? grayToRgb(stepImg) : cloneImg(stepImg);
                setDebugSteps((prev) => [...prev, { label, img: vis, timing }]);
                await new Promise<void>((r) => setTimeout(r, 0));
              }
            : undefined,
        });

        if (genCounter.current !== gen) return; // stale

        setTimings(result.timings);
        setTotalMs(result.totalMs);

        const blob = await imgToBlob(result.img);
        if (genCounter.current !== gen) return;

        const url = URL.createObjectURL(blob);
        resultUrlRef.current = url;
        setResultUrl(url);
        setState("done");

        const thumbImg = resizeImg(
          result.img,
          150,
          Math.round((150 * result.img.h) / result.img.w),
        );
        const thumbBlob = await imgToBlob(thumbImg);
        if (genCounter.current !== gen) return;
        await saveGeneration(currentParams, blob, thumbBlob, result.totalMs);
        setHistoryRefresh((n) => n + 1);
      } catch (err) {
        if (genCounter.current !== gen) return;
        console.error(err);
        setError(err instanceof Error ? err.message : String(err));
        setState("idle");
      }
    },
    [cleanup],
  );

  /** Background regeneration: keeps current output visible, swaps when ready */
  const regenerateInBackground = useCallback(async (currentParams: SprayParams) => {
    if (!sourceImgRef.current) return;
    setIsRegenerating(true);

    const gen = ++genCounter.current;

    try {
      const result = await sprayAnything(sourceImgRef.current, currentParams, {});

      if (genCounter.current !== gen) return; // stale, newer one in flight

      setTimings(result.timings);
      setTotalMs(result.totalMs);

      const blob = await imgToBlob(result.img);
      if (genCounter.current !== gen) return;

      // Swap the image
      if (resultUrlRef.current) URL.revokeObjectURL(resultUrlRef.current);
      const url = URL.createObjectURL(blob);
      resultUrlRef.current = url;
      setResultUrl(url);
      setDebugSteps([]);
      setState("done");
    } catch (err) {
      if (genCounter.current !== gen) return;
      console.error(err);
    } finally {
      if (genCounter.current === gen) setIsRegenerating(false);
    }
  }, []);

  // Live preview: regenerate on param change (debounced)
  const liveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevParamsRef = useRef<SprayParams>(params);

  useEffect(() => {
    if (!livePreview || !sourceImgRef.current || state === "idle") return;
    if (params === prevParamsRef.current) return;
    prevParamsRef.current = params;

    // Debounce: wait 100ms after last change before regenerating
    if (liveTimerRef.current) clearTimeout(liveTimerRef.current);
    liveTimerRef.current = setTimeout(() => {
      void regenerateInBackground(params);
    }, 100);

    return () => {
      if (liveTimerRef.current) clearTimeout(liveTimerRef.current);
    };
  }, [params, livePreview, state, regenerateInBackground]);

  const handleFile = useCallback(
    async (file: File) => {
      const img = await loadImageFromFile(file);
      sourceImgRef.current = img;
      await processImage(img, params, debug);
    },
    [processImage, params, debug],
  );

  const handleSample = useCallback(
    async (url: string) => {
      const img = await loadImageFromUrl(url);
      sourceImgRef.current = img;
      await processImage(img, params, debug);
    },
    [processImage, params, debug],
  );

  const handleRegenerate = useCallback(() => {
    if (!sourceImgRef.current) return;
    void processImage(sourceImgRef.current, params, debug);
  }, [processImage, params, debug]);

  const handleRestore = useCallback(
    (_params: SprayParams, imageUrl: string) => {
      cleanup();
      setParams(_params);
      resultUrlRef.current = imageUrl;
      setResultUrl(imageUrl);
      setDebugSteps([]);
      setTimings([]);
      setState("done");
    },
    [cleanup],
  );

  const handleReset = useCallback(() => {
    cleanup();
    genCounter.current++;
    sourceImgRef.current = null;
    setState("idle");
    setResultUrl(null);
    setDebugSteps([]);
    setTimings([]);
    setError(null);
    setIsRegenerating(false);
  }, [cleanup]);

  const handleDownload = useCallback(() => {
    if (!resultUrl) return;
    const a = document.createElement("a");
    a.href = resultUrl;
    a.download = "sprayed.png";
    a.click();
  }, [resultUrl]);

  const isProcessing = state === "processing";
  const hasSource = sourceImgRef.current !== null;

  return (
    <div className={`layout ${sidebarOpen ? "sidebar-open" : ""}`}>
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>Parameters</h2>
          <button className="sidebar-close" onClick={() => setSidebarOpen(false)}>
            &times;
          </button>
        </div>
        <div className="sidebar-body">
          <ParameterEditor params={params} onChange={setParams} disabled={isProcessing} />
        </div>
        <div className="sidebar-footer">
          <label className="live-preview-toggle">
            <input
              type="checkbox"
              checked={livePreview}
              onChange={(e) => setLivePreview(e.target.checked)}
            />
            Live preview
            {isRegenerating && <span className="live-indicator" />}
          </label>
          {hasSource && !livePreview && (
            <button
              className="btn-primary btn-full"
              disabled={isProcessing}
              onClick={handleRegenerate}
            >
              {isProcessing ? "Processing..." : "Regenerate"}
            </button>
          )}
        </div>
      </aside>

      <main className="main">
        <div className="main-content">
          <h1>Spray Anything</h1>
          <p className="subtitle">Drop an image to apply a spray paint effect</p>

          <div className="toolbar">
            <button className="btn-secondary" onClick={() => setSidebarOpen((o) => !o)}>
              {sidebarOpen ? "Hide" : "Show"} Parameters
            </button>
            <label className="debug-toggle">
              <input
                type="checkbox"
                checked={debug}
                onChange={(e) => setDebug(e.target.checked)}
                disabled={isProcessing}
              />
              Debug steps
            </label>
          </div>

          {state === "idle" && (
            <DropZone
              onFile={(f) => void handleFile(f)}
              onSample={(u) => void handleSample(u)}
              disabled={isProcessing}
            />
          )}

          {isProcessing && !resultUrl && (
            <div className="progress-area">
              <div className="progress-bar-track">
                <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
              </div>
              <p className="progress-label">{progressLabel}</p>
            </div>
          )}

          {error && <p className="error-msg">Error: {error}</p>}

          {resultUrl && (
            <div className="output-area">
              <div className="output-img-wrapper">
                <img
                  className={`output-img checkerboard ${isRegenerating ? "regenerating" : ""}`}
                  src={resultUrl}
                  alt="Spray paint result"
                />
                {isRegenerating && <div className="regenerating-overlay">Updating...</div>}
              </div>
              <div className="output-meta">
                Total: {(totalMs / 1000).toFixed(2)}s
                {timings.length > 0 && <span> ({timings.length} steps)</span>}
              </div>
              {timings.length > 0 && (
                <details className="timing-details">
                  <summary>Step timings</summary>
                  <div className="timing-list">
                    {timings.map((t, i) => (
                      <div key={i} className="timing-row">
                        <span>{t.label}</span>
                        <span className="timing-value">{t.durationMs.toFixed(0)}ms</span>
                      </div>
                    ))}
                  </div>
                </details>
              )}
              <div className="output-actions">
                <button className="btn-primary" onClick={handleDownload}>
                  Download
                </button>
                {!livePreview && (
                  <button
                    className="btn-secondary"
                    onClick={handleRegenerate}
                    disabled={isProcessing}
                  >
                    Regenerate
                  </button>
                )}
                <button className="btn-secondary" onClick={handleReset}>
                  New image
                </button>
              </div>
            </div>
          )}

          {debug && <DebugGrid steps={debugSteps} />}

          <HistoryPanel onRestore={handleRestore} refreshKey={historyRefresh} />
        </div>
      </main>
    </div>
  );
}
