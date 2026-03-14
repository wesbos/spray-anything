import { useState, useCallback, type DragEvent } from "react";

interface DropZoneProps {
  onFile: (file: File) => void;
  onSample: (url: string) => void;
  disabled?: boolean;
}

const SAMPLES = [{ src: "/samples/syntax-logo.png", alt: "Syntax logo" }];

export function DropZone({ onFile, onSample, disabled }: DropZoneProps) {
  const [dragover, setDragover] = useState(false);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragover(false);
      const file = e.dataTransfer?.files[0];
      if (file) onFile(file);
    },
    [onFile],
  );

  return (
    <div className="drop-zone-wrapper">
      <div
        className={`drop-zone ${dragover ? "dragover" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragover(true);
        }}
        onDragLeave={() => setDragover(false)}
        onDrop={handleDrop}
        onClick={() => {
          if (disabled) return;
          const input = document.createElement("input");
          input.type = "file";
          input.accept = "image/*";
          input.onchange = () => {
            const file = input.files?.[0];
            if (file) onFile(file);
          };
          input.click();
        }}
      >
        <p>Drag & drop an image here</p>
        <p>or click to choose a file</p>
      </div>

      <div className="samples">
        <p className="samples-label">Or try a sample:</p>
        <div className="samples-list">
          {SAMPLES.map((s) => (
            <button
              key={s.src}
              className="sample-btn"
              disabled={disabled}
              onClick={() => onSample(s.src)}
            >
              <img src={s.src} alt={s.alt} />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
