import { type Img, createImg, clamp } from "./image.ts";
import { gpuGaussianBlur } from "./webgpu.ts";

/** Convert an Img (RGB Float32) to an OffscreenCanvas with RGBA pixels */
function imgToCanvas(img: Img): OffscreenCanvas {
  const canvas = new OffscreenCanvas(img.w, img.h);
  const ctx = canvas.getContext("2d")!;
  const imageData = ctx.createImageData(img.w, img.h);
  const d = imageData.data;
  const n = img.w * img.h;

  if (img.c === 3) {
    for (let i = 0; i < n; i++) {
      d[i * 4] = clamp(Math.round(img.data[i * 3]!));
      d[i * 4 + 1] = clamp(Math.round(img.data[i * 3 + 1]!));
      d[i * 4 + 2] = clamp(Math.round(img.data[i * 3 + 2]!));
      d[i * 4 + 3] = 255;
    }
  } else {
    for (let i = 0; i < n; i++) {
      const v = clamp(Math.round(img.data[i]!));
      d[i * 4] = v;
      d[i * 4 + 1] = v;
      d[i * 4 + 2] = v;
      d[i * 4 + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

/** Convert canvas RGBA data back to an RGB Float32 Img */
function canvasToImg(canvas: OffscreenCanvas): Img {
  const ctx = canvas.getContext("2d")!;
  const w = canvas.width;
  const h = canvas.height;
  const imageData = ctx.getImageData(0, 0, w, h);
  const d = imageData.data;
  const img = createImg(w, h, 3);
  const n = w * h;
  for (let i = 0; i < n; i++) {
    img.data[i * 3] = d[i * 4]!;
    img.data[i * 3 + 1] = d[i * 4 + 1]!;
    img.data[i * 3 + 2] = d[i * 4 + 2]!;
  }
  return img;
}

/** Draw a bitmap onto a white background, stripping alpha like sharp's removeAlpha() */
function bitmapToWhiteBgCanvas(bitmap: ImageBitmap): OffscreenCanvas {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d")!;
  // Fill white first so transparent areas composite to white, not black
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, bitmap.width, bitmap.height);
  ctx.drawImage(bitmap, 0, 0);
  return canvas;
}

/** Load an image from a File (drag-drop / file picker) */
export async function loadImageFromFile(file: File): Promise<Img> {
  const bitmap = await createImageBitmap(file);
  const canvas = bitmapToWhiteBgCanvas(bitmap);
  bitmap.close();
  return canvasToImg(canvas);
}

/** Load an image from a URL (for SprayMap) */
export async function loadImageFromUrl(url: string): Promise<Img> {
  const resp = await fetch(url);
  const blob = await resp.blob();
  const bitmap = await createImageBitmap(blob);
  const canvas = bitmapToWhiteBgCanvas(bitmap);
  bitmap.close();
  return canvasToImg(canvas);
}

/** Gaussian blur — GPU separable blur with Canvas fallback */
export async function gaussianBlur(img: Img, sigma: number): Promise<Img> {
  const gpu = await gpuGaussianBlur(img, sigma);
  if (gpu) return gpu;
  return gaussianBlurCanvas(img, sigma);
}

function gaussianBlurCanvas(img: Img, sigma: number): Img {
  // Pad the canvas by the blur radius to avoid edge darkening from
  // transparency bleed. CSS blur() takes sigma directly in px.
  const pad = Math.ceil(sigma * 3);
  const srcCanvas = imgToCanvas(img);
  const padW = img.w + pad * 2;
  const padH = img.h + pad * 2;

  const dst = new OffscreenCanvas(padW, padH);
  const ctx = dst.getContext("2d")!;

  // Fill with edge-extended content: draw the image offset by pad,
  // then tile the edges by stretching 1px strips
  ctx.drawImage(srcCanvas, pad, pad);
  // Top edge
  ctx.drawImage(srcCanvas, 0, 0, img.w, 1, pad, 0, img.w, pad);
  // Bottom edge
  ctx.drawImage(srcCanvas, 0, img.h - 1, img.w, 1, pad, pad + img.h, img.w, pad);
  // Left edge
  ctx.drawImage(srcCanvas, 0, 0, 1, img.h, 0, pad, pad, img.h);
  // Right edge
  ctx.drawImage(srcCanvas, img.w - 1, 0, 1, img.h, pad + img.w, pad, pad, img.h);
  // Corners
  ctx.drawImage(srcCanvas, 0, 0, 1, 1, 0, 0, pad, pad);
  ctx.drawImage(srcCanvas, img.w - 1, 0, 1, 1, pad + img.w, 0, pad, pad);
  ctx.drawImage(srcCanvas, 0, img.h - 1, 1, 1, 0, pad + img.h, pad, pad);
  ctx.drawImage(srcCanvas, img.w - 1, img.h - 1, 1, 1, pad + img.w, pad + img.h, pad, pad);

  const blurred = new OffscreenCanvas(padW, padH);
  const bCtx = blurred.getContext("2d")!;
  bCtx.filter = `blur(${sigma}px)`;
  bCtx.drawImage(dst, 0, 0);

  // Crop back to original size
  const crop = new OffscreenCanvas(img.w, img.h);
  const cCtx = crop.getContext("2d")!;
  cCtx.drawImage(blurred, -pad, -pad);
  return canvasToImg(crop);
}

/** Resize using Canvas drawImage with high-quality smoothing */
export function resizeImg(img: Img, w: number, h: number): Img {
  const src = imgToCanvas(img);
  const dst = new OffscreenCanvas(w, h);
  const ctx = dst.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, 0, 0, w, h);
  return canvasToImg(dst);
}

/** Convert an Img to a Blob for download/preview */
export async function imgToBlob(img: Img): Promise<Blob> {
  const canvas = imgToCanvas(img);
  return canvas.convertToBlob({ type: "image/png" });
}
