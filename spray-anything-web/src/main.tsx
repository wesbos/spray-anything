import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.tsx";
import { warmUpGpu } from "./webgpu.ts";
import "./style.css";

// Pre-compile GPU shaders while the user picks an image
void warmUpGpu();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
