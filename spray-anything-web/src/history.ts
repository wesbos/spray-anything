import type { SprayParams } from "./params.ts";

export interface HistoryEntry {
  id: number;
  timestamp: number;
  params: SprayParams;
  imageBlob: Blob;
  thumbnailBlob: Blob;
  totalTimeMs: number;
}

const DB_NAME = "spray-anything";
const DB_VERSION = 1;
const STORE_NAME = "generations";

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id", autoIncrement: true });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function saveGeneration(
  params: SprayParams,
  imageBlob: Blob,
  thumbnailBlob: Blob,
  totalTimeMs: number,
): Promise<number> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const entry: Omit<HistoryEntry, "id"> = {
      timestamp: Date.now(),
      params,
      imageBlob,
      thumbnailBlob,
      totalTimeMs,
    };
    const req = store.add(entry);
    req.onsuccess = () => resolve(req.result as number);
    req.onerror = () => reject(req.error);
  });
}

export async function loadHistory(): Promise<HistoryEntry[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const req = store.getAll();
    req.onsuccess = () => {
      const entries = req.result as HistoryEntry[];
      entries.sort((a, b) => b.timestamp - a.timestamp);
      resolve(entries);
    };
    req.onerror = () => reject(req.error);
  });
}

export async function deleteGeneration(id: number): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const req = store.delete(id);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

export async function clearHistory(): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const req = store.clear();
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}
