import { useState, useEffect, useCallback } from "react";
import { type HistoryEntry, loadHistory, deleteGeneration, clearHistory } from "../history.ts";
import type { SprayParams } from "../params.ts";

interface HistoryPanelProps {
  onRestore: (params: SprayParams, imageUrl: string) => void;
  refreshKey: number;
}

export function HistoryPanel({ onRestore, refreshKey }: HistoryPanelProps) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [thumbUrls, setThumbUrls] = useState<Map<number, string>>(new Map());
  const [open, setOpen] = useState(false);

  const refresh = useCallback(() => {
    void loadHistory().then((e) => {
      setEntries(e);
      setThumbUrls((prev) => {
        const next = new Map(prev);
        for (const entry of e) {
          if (!next.has(entry.id)) {
            next.set(entry.id, URL.createObjectURL(entry.thumbnailBlob));
          }
        }
        // Clean up removed entries
        for (const id of prev.keys()) {
          if (!e.some((x) => x.id === id)) {
            URL.revokeObjectURL(prev.get(id)!);
            next.delete(id);
          }
        }
        return next;
      });
    });
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh, refreshKey]);

  // Cleanup blob urls on unmount
  useEffect(() => {
    return () => {
      for (const url of thumbUrls.values()) URL.revokeObjectURL(url);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- cleanup only
  }, []);

  if (!open) {
    return (
      <button className="history-toggle" onClick={() => setOpen(true)}>
        History ({entries.length})
      </button>
    );
  }

  return (
    <div className="history-panel">
      <div className="history-header">
        <h2>History</h2>
        <div className="history-actions">
          {entries.length > 0 && (
            <button
              className="btn-small btn-danger"
              onClick={() => {
                void clearHistory().then(refresh);
              }}
            >
              Clear all
            </button>
          )}
          <button className="btn-small" onClick={() => setOpen(false)}>
            Close
          </button>
        </div>
      </div>

      {entries.length === 0 ? (
        <p className="history-empty">No generations yet</p>
      ) : (
        <div className="history-grid">
          {entries.map((entry) => (
            <div key={entry.id} className="history-item">
              <img
                src={thumbUrls.get(entry.id)}
                alt={`Generation ${entry.id}`}
                onClick={() => {
                  const url = URL.createObjectURL(entry.imageBlob);
                  onRestore(entry.params, url);
                }}
              />
              <div className="history-meta">
                <span>{new Date(entry.timestamp).toLocaleString()}</span>
                <span>{(entry.totalTimeMs / 1000).toFixed(1)}s</span>
              </div>
              <button
                className="btn-small btn-danger"
                onClick={() => {
                  void deleteGeneration(entry.id).then(refresh);
                }}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
