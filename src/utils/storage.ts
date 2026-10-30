import { EvaluationResults, ReviewItem, Theme, UploadedDatasetRecord } from '../types';
import { SerializedEngineState } from '../ml/engine';

const DB_NAME = 'SentimentEnsembleDB';
const DB_VERSION = 1;
const STORE_NAME = 'ml_sentiment_store';
const THEME_STORAGE_KEY = 'sentiment_app_theme';

export interface PersistedMetadata {
  modelVersion: number;
  datasetsCount: number;
  lastTrainedTime: string | null;
  isDemo: boolean;
  sourceName: string;
  lastBatchAddedCount?: number;
  lastBatchDuplicatesRemoved?: number;
  uploadedDatasets?: UploadedDatasetRecord[];
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof window === 'undefined' || !window.indexedDB) {
      reject(new Error('IndexedDB not supported in this environment'));
      return;
    }

    const request = window.indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Saves a key-value record to IndexedDB.
 */
export async function saveToIndexedDB<T>(key: string, value: T): Promise<void> {
  try {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const req = store.put(value, key);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    console.warn(`IndexedDB save failed for ${key}, falling back to localStorage if small:`, err);
    try {
      if (typeof window !== 'undefined') {
        // Fallback for smaller values
        if (key !== 'accumulated_reviews' || JSON.stringify(value).length < 2000000) {
          localStorage.setItem(`idb_fallback_${key}`, JSON.stringify(value));
        }
      }
    } catch (e) {
      console.warn('Storage fallback failed:', e);
    }
  }
}

/**
 * Retrieves a key-value record from IndexedDB.
 */
export async function getFromIndexedDB<T>(key: string): Promise<T | null> {
  try {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const req = store.get(key);
      req.onsuccess = () => resolve(req.result ?? null);
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    console.warn(`IndexedDB get failed for ${key}, checking fallback:`, err);
    try {
      if (typeof window !== 'undefined') {
        const item = localStorage.getItem(`idb_fallback_${key}`);
        return item ? JSON.parse(item) : null;
      }
    } catch {
      return null;
    }
    return null;
  }
}

/**
 * Persists the entire training state: accumulated dataset, metadata, evaluation results, and trained engine state.
 */
export async function persistAllState(
  reviews: ReviewItem[],
  metadata: PersistedMetadata,
  evaluation: EvaluationResults | null,
  engineState?: SerializedEngineState | null
): Promise<void> {
  const tasks: Promise<void>[] = [
    saveToIndexedDB('accumulated_reviews', reviews),
    saveToIndexedDB('dataset_metadata', metadata),
    saveToIndexedDB('evaluation_results', evaluation)
  ];
  if (engineState !== undefined) {
    tasks.push(saveToIndexedDB('engine_pipeline_state', engineState));
  }
  await Promise.all(tasks);
}

/**
 * Loads the persisted state from IndexedDB.
 */
export async function loadPersistedState(): Promise<{
  reviews: ReviewItem[] | null;
  metadata: PersistedMetadata | null;
  evaluation: EvaluationResults | null;
  engineState: SerializedEngineState | null;
}> {
  try {
    const [reviews, metadata, evaluation, engineState] = await Promise.all([
      getFromIndexedDB<ReviewItem[]>('accumulated_reviews'),
      getFromIndexedDB<PersistedMetadata>('dataset_metadata'),
      getFromIndexedDB<EvaluationResults>('evaluation_results'),
      getFromIndexedDB<SerializedEngineState>('engine_pipeline_state')
    ]);

    return { reviews, metadata, evaluation, engineState };
  } catch (err) {
    console.error('Failed to load persisted state:', err);
    return { reviews: null, metadata: null, evaluation: null, engineState: null };
  }
}

/**
 * Theme persistence (localStorage)
 */
export function getStoredTheme(): Theme {
  if (typeof window === 'undefined') return 'dark';
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === 'light' || saved === 'dark') {
      return saved;
    }
  } catch {
    // ignore
  }
  return 'dark'; // default to clean dark theme
}

export function saveStoredTheme(theme: Theme): void {
  try {
    if (typeof window !== 'undefined') {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    }
  } catch {
    // ignore
  }
}
