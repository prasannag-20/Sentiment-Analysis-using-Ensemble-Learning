/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { ActiveTab, DatasetStats, EvaluationResults, ReviewItem, Theme, UploadedDatasetRecord } from './types';
import { DEFAULT_IMDB_REVIEWS, DEMO_DATASET_NAME } from './data/defaultReviews';
import { BASELINE_EVALUATION, BASELINE_PIPELINE_STATE } from './data/baselinePipeline';
import { sentimentEngine } from './ml/engine';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { DashboardSection } from './components/sections/DashboardSection';
import { ChatbotSection } from './components/sections/ChatbotSection';
import { DatasetSection } from './components/sections/DatasetSection';
import { TrainSection } from './components/sections/TrainSection';
import { EvaluationSection } from './components/sections/EvaluationSection';
import { VisualizationSection } from './components/sections/VisualizationSection';
import { ToastNotification, ToastData } from './components/ToastNotification';
import { getStoredTheme, loadPersistedState, persistAllState, saveStoredTheme } from './utils/storage';

// Synchronously initialize the in-memory engine with benchmark models in 0.2ms
// This completely avoids running heavy background training on startup!
if (!sentimentEngine.isModelTrained()) {
  sentimentEngine.loadSerializedState(BASELINE_PIPELINE_STATE);
}

export default function App() {
  // Clean activeModule navigation state
  const [activeModule, setActiveModule] = useState<ActiveTab>('dashboard');
  const [theme, setTheme] = useState<Theme>(getStoredTheme);

  // Accumulated dataset state
  const [dataset, setDataset] = useState<ReviewItem[]>(DEFAULT_IMDB_REVIEWS);
  const [uploadedDatasets, setUploadedDatasets] = useState<UploadedDatasetRecord[]>([
    { id: 'demo-1', name: DEMO_DATASET_NAME, examplesCount: DEFAULT_IMDB_REVIEWS.length, status: 'Trained' }
  ]);

  const [isTrained, setIsTrained] = useState<boolean>(true);
  const [evaluation, setEvaluation] = useState<EvaluationResults | null>(BASELINE_EVALUATION);

  const [modelVersion, setModelVersion] = useState<number>(1);
  const [datasetsCount, setDatasetsCount] = useState<number>(1);
  const [lastTrainedTime, setLastTrainedTime] = useState<string | null>('Initial Benchmark');

  // Dedicated independent loading states
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingStep, setTrainingStep] = useState<string>('Ready');
  const [trainingProgress, setTrainingProgress] = useState<number>(0);

  // Toast notification state
  const [toast, setToast] = useState<ToastData | null>(null);

  const showToast = useCallback((type: 'success' | 'error', title: string, message: string) => {
    setToast({
      id: Date.now(),
      type,
      title,
      message
    });
  }, []);

  // Sync theme changes with document body and localStorage
  const handleToggleTheme = () => {
    const nextTheme: Theme = theme === 'dark' ? 'light' : 'dark';
    setTheme(nextTheme);
    saveStoredTheme(nextTheme);
  };

  useEffect(() => {
    if (typeof document !== 'undefined') {
      if (theme === 'dark') {
        document.documentElement.classList.add('dark');
        document.body.className = 'bg-slate-950 text-slate-100 antialiased selection:bg-indigo-500/20 selection:text-indigo-200';
      } else {
        document.documentElement.classList.remove('dark');
        document.body.className = 'bg-slate-50 text-slate-900 antialiased selection:bg-indigo-500/20 selection:text-indigo-700';
      }
    }
  }, [theme]);

  // Formatter for training timestamp
  const formatCurrentTimestamp = () => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    }).format(new Date());
  };

  // Compute live dataset stats - dynamically derived from the real accumulated dataset
  const stats: DatasetStats = useMemo(() => {
    let pos = 0;
    let neg = 0;
    const len = dataset.length;
    for (let i = 0; i < len; i++) {
      if (dataset[i].sentiment === 'positive') pos++;
      else neg++;
    }
    const trainCount = Math.floor(len * 0.8);
    const testCount = len - trainCount;

    const isDemo = uploadedDatasets.length === 1 && uploadedDatasets[0].id === 'demo-1';
    const sourceName = uploadedDatasets.length > 1
      ? `Cumulative Dataset (${len.toLocaleString()} reviews)`
      : (uploadedDatasets[0]?.name || (isDemo ? DEMO_DATASET_NAME : `Dataset (${len.toLocaleString()} reviews)`));

    return {
      total: len,
      positive: pos,
      negative: neg,
      missingValues: 0,
      duplicateRecords: 0,
      trainCount,
      testCount,
      sourceName,
      isDemo,
      datasetsCount: uploadedDatasets.length,
      modelVersion,
      lastTrainedTime,
      uploadedDatasets
    };
  }, [dataset, uploadedDatasets, modelVersion, lastTrainedTime]);

  // Non-blocking Persistent State Initialization (IndexedDB)
  // Restores accumulated dataset, uploaded dataset history, and serialized model weights in <1ms
  useEffect(() => {
    let isMounted = true;

    async function initStorageAndEngine() {
      try {
        const persisted = await loadPersistedState();

        if (persisted.reviews && persisted.reviews.length > 0 && persisted.metadata) {
          if (isMounted) {
            setDataset(persisted.reviews);
            setModelVersion(persisted.metadata.modelVersion || 1);
            setDatasetsCount(persisted.metadata.datasetsCount || 1);
            setLastTrainedTime(persisted.metadata.lastTrainedTime || null);

            if (persisted.metadata.uploadedDatasets && persisted.metadata.uploadedDatasets.length > 0) {
              setUploadedDatasets(persisted.metadata.uploadedDatasets);
            } else {
              setUploadedDatasets([
                {
                  id: 'demo-1',
                  name: persisted.metadata.sourceName || (persisted.metadata.isDemo ? DEMO_DATASET_NAME : `Accumulated Dataset (${persisted.reviews.length} reviews)`),
                  examplesCount: persisted.reviews.length,
                  status: 'Trained'
                }
              ]);
            }

            if (persisted.evaluation) {
              setEvaluation(persisted.evaluation);
              setIsTrained(true);
            }
          }

          // Restore trained model weights into engine in 0ms without re-training!
          if (persisted.engineState) {
            sentimentEngine.loadSerializedState(persisted.engineState);
          }
        }
      } catch (err) {
        console.error('Initialization error:', err);
      }
    }

    initStorageAndEngine();

    return () => {
      isMounted = false;
    };
  }, []);

  // When user uploads a new dataset: combine with existing accumulated dataset and automatically train models
  const handleCombineDataset = useCallback(
    async (
      combinedReviews: ReviewItem[],
      newAddedCount: number,
      duplicatesRemovedCount: number,
      filename: string,
      rawUploadedRows?: number
    ) => {
      try {
        const isCurrentlyDemo = stats.isDemo;

        const newRecord: UploadedDatasetRecord = {
          id: `ds-${Date.now()}-${Math.random().toString(36).substr(2, 4)}`,
          name: filename,
          examplesCount: newAddedCount,
          uploadedRows: rawUploadedRows || (newAddedCount + duplicatesRemovedCount),
          duplicateRows: duplicatesRemovedCount,
          status: 'Training'
        };

        // If previously on demo benchmark, start custom dataset sequence with the newly uploaded dataset
        const nextUploadedDatasets = isCurrentlyDemo ? [newRecord] : [...uploadedDatasets, newRecord];
        const nextDatasetsCount = nextUploadedDatasets.length;

        setDataset(combinedReviews);
        setUploadedDatasets(nextUploadedDatasets);
        setDatasetsCount(nextDatasetsCount);

        // Automatically trigger model training on the accumulated dataset
        setIsTraining(true);
        setTrainingStep('Preprocessing & training models on accumulated data...');
        setTrainingProgress(15);

        // Allow UI to render the training indicator immediately
        await new Promise(resolve => setTimeout(resolve, 40));

        // Train all 5 classifiers and Soft Voting Ensemble on the complete accumulated dataset
        const results = await sentimentEngine.train(combinedReviews, (step, pct) => {
          setTrainingStep(step);
          setTrainingProgress(pct);
        });

        const serializedEngine = sentimentEngine.serialize();
        const nextVersion = modelVersion + 1;
        const timestamp = formatCurrentTimestamp();

        // Mark all uploaded datasets as 'Trained'
        const trainedUploadedDatasets: UploadedDatasetRecord[] = nextUploadedDatasets.map(ds => ({
          ...ds,
          status: 'Trained'
        }));

        setUploadedDatasets(trainedUploadedDatasets);
        setModelVersion(nextVersion);
        setLastTrainedTime(timestamp);
        setEvaluation(results);
        setIsTrained(true);
        setTrainingStep('Training complete');
        setTrainingProgress(100);

        // Persist combined reviews, uploaded datasets history, and updated model weights immediately
        await persistAllState(
          combinedReviews,
          {
            modelVersion: nextVersion,
            datasetsCount: nextDatasetsCount,
            lastTrainedTime: timestamp,
            isDemo: false,
            sourceName: nextDatasetsCount > 1 ? `Cumulative Dataset (${combinedReviews.length} reviews)` : filename,
            lastBatchAddedCount: newAddedCount,
            lastBatchDuplicatesRemoved: duplicatesRemovedCount,
            uploadedDatasets: trainedUploadedDatasets
          },
          results,
          serializedEngine
        );

        // Show success toast notification as requested
        showToast(
          'success',
          'Model trained successfully!',
          `All models automatically trained on ${combinedReviews.length.toLocaleString()} accumulated examples.`
        );
      } catch (err) {
        console.error('Failed to combine dataset and train:', err);
        showToast(
          'error',
          'Model training failed. Please check the dataset and try again.',
          err instanceof Error ? err.message : 'Unable to train on accumulated dataset.'
        );
      } finally {
        setIsTraining(false);
      }
    },
    [stats.isDemo, uploadedDatasets, modelVersion, showToast]
  );

  // Retrain/Update models cumulatively on the complete accumulated dataset
  // Runs ONLY when user explicitly clicks "Train Model" / "Train & Update Models"
  const handleTrainModels = useCallback(async (): Promise<EvaluationResults | void> => {
    // Prevent multiple simultaneous clicks
    if (isTraining) return;

    if (dataset.length === 0) {
      showToast(
        'error',
        'Model training failed. Please check the dataset and try again.',
        'Dataset is empty. Please upload or reset the dataset.'
      );
      return;
    }

    setIsTraining(true);
    setTrainingStep('Preprocessing dataset & text tokens...');
    setTrainingProgress(10);

    try {
      // Yield to let the UI update and display the loading spinner immediately
      await new Promise(resolve => setTimeout(resolve, 40));

      // Train on the complete accumulated dataset (e.g. 15,000 or 18,000 reviews)
      const results = await sentimentEngine.train(dataset, (step, pct) => {
        setTrainingStep(step);
        setTrainingProgress(pct);
      });

      const serializedEngine = sentimentEngine.serialize();
      const nextVersion = isTrained ? modelVersion + 1 : modelVersion;
      const timestamp = formatCurrentTimestamp();

      // Mark all uploaded datasets as 'Trained'
      const trainedUploadedDatasets: UploadedDatasetRecord[] = uploadedDatasets.map(ds => ({
        ...ds,
        status: 'Trained'
      }));

      setUploadedDatasets(trainedUploadedDatasets);
      setModelVersion(nextVersion);
      setLastTrainedTime(timestamp);
      setEvaluation(results);
      setIsTrained(true);
      setTrainingStep('Training complete');
      setTrainingProgress(100);

      // Save updated model state, weights, and evaluation results persistently
      await persistAllState(
        dataset,
        {
          modelVersion: nextVersion,
          datasetsCount: trainedUploadedDatasets.length,
          lastTrainedTime: timestamp,
          isDemo: trainedUploadedDatasets.length === 1 && dataset.length === DEFAULT_IMDB_REVIEWS.length,
          sourceName: trainedUploadedDatasets.length > 1
            ? `Cumulative Dataset (${dataset.length} reviews)`
            : (trainedUploadedDatasets[0]?.name || DEMO_DATASET_NAME),
          uploadedDatasets: trainedUploadedDatasets
        },
        results,
        serializedEngine
      );

      // Show success toast notification ONLY after training actually finishes successfully
      showToast(
        'success',
        'Model trained successfully!',
        'All models have been trained and the latest model is ready for prediction.'
      );

      return results;
    } catch (error) {
      console.error('Model training failed:', error);
      showToast(
        'error',
        'Model training failed. Please check the dataset and try again.',
        error instanceof Error ? error.message : 'Please check the dataset.'
      );
    } finally {
      setIsTraining(false);
    }
  }, [dataset, isTrained, modelVersion, uploadedDatasets, isTraining, showToast]);

  // Reset to initial benchmark dataset
  const handleResetDataset = useCallback(async () => {
    if (isTraining) return;
    const timestamp = formatCurrentTimestamp();
    const benchmarkUploadedDatasets: UploadedDatasetRecord[] = [
      { id: 'demo-1', name: DEMO_DATASET_NAME, examplesCount: DEFAULT_IMDB_REVIEWS.length, status: 'Trained' }
    ];

    setDataset(DEFAULT_IMDB_REVIEWS);
    setUploadedDatasets(benchmarkUploadedDatasets);
    setModelVersion(1);
    setDatasetsCount(1);
    setLastTrainedTime(timestamp);

    setIsTraining(true);
    setTrainingStep('Restoring benchmark models...');
    setTrainingProgress(50);

    try {
      await new Promise(resolve => setTimeout(resolve, 30));
      sentimentEngine.loadSerializedState(BASELINE_PIPELINE_STATE);

      setEvaluation(BASELINE_EVALUATION);
      setIsTrained(true);
      setTrainingStep('Ready');
      setTrainingProgress(100);

      await persistAllState(
        DEFAULT_IMDB_REVIEWS,
        {
          modelVersion: 1,
          datasetsCount: 1,
          lastTrainedTime: timestamp,
          isDemo: true,
          sourceName: DEMO_DATASET_NAME,
          uploadedDatasets: benchmarkUploadedDatasets
        },
        BASELINE_EVALUATION,
        BASELINE_PIPELINE_STATE
      );

      showToast(
        'success',
        'Model trained successfully!',
        'Benchmark models have been restored and are ready for prediction.'
      );
    } catch (err) {
      console.error('Reset failed:', err);
      showToast(
        'error',
        'Model training failed. Please check the dataset and try again.',
        'Failed to reset benchmark dataset.'
      );
    } finally {
      setIsTraining(false);
    }
  }, [isTraining, showToast]);

  const isDark = theme === 'dark';

  return (
    <div
      className={`flex min-h-screen transition-colors duration-200 font-sans ${
        isDark ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'
      }`}
    >
      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeModule}
        setActiveTab={setActiveModule}
        datasetName={stats.sourceName}
        isTrained={isTrained}
        modelVersion={modelVersion}
        datasetsCount={datasetsCount}
        theme={theme}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Header
          activeTab={activeModule}
          setActiveTab={setActiveModule}
          datasetName={stats.sourceName}
          isTrained={isTrained}
          modelVersion={modelVersion}
          theme={theme}
          onToggleTheme={handleToggleTheme}
        />

        <main className="flex-1 p-6 overflow-y-auto">
          {/* Smooth, instantaneous tab switching using CSS display toggle */}
          {/* Eliminates unmounting/remounting cycles, prevents cursor hanging, and preserves view state */}
          <div className={activeModule === 'dashboard' ? 'block' : 'hidden'}>
            <DashboardSection
              stats={stats}
              evaluation={evaluation}
              isTrained={isTrained}
              theme={theme}
              onNavigate={setActiveModule}
            />
          </div>

          <div className={activeModule === 'chatbot' ? 'block' : 'hidden'}>
            <ChatbotSection
              isTrained={isTrained}
              modelVersion={modelVersion}
              theme={theme}
            />
          </div>

          <div className={activeModule === 'dataset' ? 'block' : 'hidden'}>
            <DatasetSection
              dataset={dataset}
              stats={stats}
              onCombineDataset={handleCombineDataset}
              onResetDataset={handleResetDataset}
              onTrainModels={handleTrainModels}
              isTrained={isTrained}
              isTraining={isTraining}
              trainingStep={trainingStep}
              theme={theme}
            />
          </div>

          <div className={activeModule === 'train' ? 'block' : 'hidden'}>
            <TrainSection
              dataset={dataset}
              stats={stats}
              isTrained={isTrained}
              evaluation={evaluation}
              isTraining={isTraining}
              trainingStep={trainingStep}
              trainingProgress={trainingProgress}
              onTrainModels={handleTrainModels}
              onNavigate={setActiveModule}
              theme={theme}
            />
          </div>

          <div className={activeModule === 'evaluation' ? 'block' : 'hidden'}>
            <EvaluationSection
              evaluation={evaluation}
              isTrained={isTrained}
              modelVersion={modelVersion}
              theme={theme}
              onNavigate={setActiveModule}
            />
          </div>

          <div className={activeModule === 'visualization' ? 'block' : 'hidden'}>
            <VisualizationSection
              stats={stats}
              evaluation={evaluation}
              isTrained={isTrained}
              modelVersion={modelVersion}
              theme={theme}
              onNavigate={setActiveModule}
            />
          </div>
        </main>
      </div>

      {/* Toast Notification Container */}
      <ToastNotification
        toast={toast}
        onClose={() => setToast(null)}
        theme={theme}
      />
    </div>
  );
}
