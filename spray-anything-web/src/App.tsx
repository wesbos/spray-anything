import { useState, useCallback, useRef } from "react";
import type { Img } from "./image.ts";
import { cloneImg, grayToRgb } from "./image.ts";
import { loadImageFromFile, loadImageFromUrl, imgToBlob } from "./canvas-io.ts";
import { resizeImg } from "./canvas-io.ts";
import { sprayAnything } from "./pipeline.ts";
import type { StepTiming } from "./pipeline.ts";
import { type SprayParams, DEFAULT_PARAMS } from "./params.ts";
import { saveGeneration } from "./history.ts";
import { DropZone } from "./components/DropZone.tsx";
import { ParameterEditor } from "./components/ParameterEditor.tsx";
import { DebugGrid, type DebugStep } from "./components/DebugGrid.tsx";
import { HistoryPanel } from "./components/HistoryPanel.tsx";

type AppState = "idle" | "processing" | "done";

export function App() {
  const [state, setState] = useState<AppState>("idle");
  const [params, setParams] = useState<SprayParams>(DEFAULT_PARAMS);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [debug, setDebug] = useState(false);
  const [progressPct, setProgressPct] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [debugSteps, setDebugSteps] = useState<DebugStep[]>([]);
  const [timings, setTimings] = useState<StepTiming[]>([]);
  const [totalMs, setTotalMs] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [historyRefresh, setHistoryRefresh] = useState(0);

  const resultUrlRef = useRef<string | null>(null);
  // Keep the source image so we can re-run with different params
  const sourceImgRef = useRef<Img | null>(null);

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

      try {
        const result = await sprayAnything(img, currentParams, {
          onProgress(step, pct) {
            setProgressPct(pct);
            setProgressLabel(step);
          },
          onDebugStep: currentDebug
            ? async (label, stepImg, timing) => {
                const vis = stepImg.c === 1 ? grayToRgb(stepImg) : cloneImg(stepImg);
                setDebugSteps((prev) => [...prev, { label, img: vis, timing }]);
                await new Promise<void>((r) => setTimeout(r, 0));
              }
            : undefined,
        });

        setTimings(result.timings);
        setTotalMs(result.totalMs);

        const blob = await imgToBlob(result.img);
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
        await saveGeneration(currentParams, blob, thumbBlob, result.totalMs);
        setHistoryRefresh((n) => n + 1);
      } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : String(err));
        setState("idle");
      }
    },
    [cleanup],
  );

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
    sourceImgRef.current = null;
    setState("idle");
    setResultUrl(null);
    setDebugSteps([]);
    setTimings([]);
    setError(null);
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
          {hasSource && (
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

          {isProcessing && (
            <div className="progress-area">
              <div className="progress-bar-track">
                <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
              </div>
              <p className="progress-label">{progressLabel}</p>
            </div>
          )}

          {error && <p className="error-msg">Error: {error}</p>}

          {state === "done" && resultUrl && (
            <div className="output-area">
              <img className="output-img checkerboard" src={resultUrl} alt="Spray paint result" />
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
                <button
                  className="btn-secondary"
                  onClick={handleRegenerate}
                  disabled={isProcessing}
                >
                  Regenerate
                </button>
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
