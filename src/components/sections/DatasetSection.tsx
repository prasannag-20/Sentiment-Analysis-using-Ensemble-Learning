import React, { useState, useMemo } from 'react';
import { 
  Upload, 
  Download, 
  Search, 
  RotateCcw,
  CheckCircle2, 
  AlertCircle,
  Cpu, 
  Layers, 
  Calendar,
  Sparkles,
  Database,
  Loader2
} from 'lucide-react';
import { BatchReviewItem, DatasetStats, EvaluationResults, ReviewItem, Theme } from '../../types';
import { combineAndDeduplicateReviews, downloadCSV, parseDatasetFile, parseUnlabeledFile } from '../../utils/fileParser';
import { sentimentEngine } from '../../ml/engine';
import { DEFAULT_IMDB_REVIEWS, DEMO_DATASET_NAME } from '../../data/defaultReviews';

interface DatasetSectionProps {
  dataset: ReviewItem[];
  stats: DatasetStats;
  onCombineDataset: (
    combinedReviews: ReviewItem[],
    newAddedCount: number,
    duplicatesRemovedCount: number,
    filename: string,
    rawUploadedRows?: number
  ) => void;
  onResetDataset: () => void;
  onTrainModels: () => Promise<EvaluationResults | void>;
  isTrained: boolean;
  isTraining: boolean;
  trainingStep: string;
  theme: Theme;
}

export const DatasetSection: React.FC<DatasetSectionProps> = ({
  dataset,
  stats,
  onCombineDataset,
  onResetDataset,
  onTrainModels,
  isTrained,
  isTraining,
  trainingStep,
  theme
}) => {
  const isDark = theme === 'dark';

  // Upload States
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [lastUploadInfo, setLastUploadInfo] = useState<{
    filename: string;
    uploadedRows: number;
    newRecords: number;
    duplicatesRemoved: number;
    totalAccumulated: number;
  } | null>(null);

  // Table Search and Pagination
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 8;

  // Batch Prediction States
  const [batchReviews, setBatchReviews] = useState<BatchReviewItem[]>([]);
  const [isProcessingBatch, setIsProcessingBatch] = useState(false);
  const [batchError, setBatchError] = useState<string | null>(null);

  // Handle Training Dataset Upload
  const handleDatasetFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadError(null);

    const result = await parseDatasetFile(file);
    setIsUploading(false);

    if (!result.success) {
      setUploadError(result.error || 'Failed to parse file.');
      return;
    }

    // Cumulative dataset combination:
    // If currently on demo benchmark, replace demo dataset so first custom dataset starts clean.
    // If already on custom data, combine new unique reviews with existing accumulated dataset.
    const isCurrentlyDemo = stats.isDemo;
    const baseReviews = isCurrentlyDemo ? [] : dataset;

    const { combined, newAddedCount, duplicatesRemovedCount } = combineAndDeduplicateReviews(
      baseReviews,
      result.reviews
    );

    const totalDuplicates = duplicatesRemovedCount + result.duplicateCount;
    const rawUploadedRows = result.totalRawRows || result.reviews.length;

    setLastUploadInfo({
      filename: file.name,
      uploadedRows: rawUploadedRows,
      newRecords: newAddedCount,
      duplicatesRemoved: totalDuplicates,
      totalAccumulated: combined.length
    });

    onCombineDataset(
      combined,
      newAddedCount,
      totalDuplicates,
      file.name,
      rawUploadedRows
    );
    setCurrentPage(1);
  };

  // Filtered Reviews for Preview Table - Memoized to prevent UI stutter
  const filteredReviews = useMemo(() => {
    if (!searchTerm.trim()) return dataset;
    const term = searchTerm.toLowerCase();
    return dataset.filter(r =>
      r.review.toLowerCase().includes(term) ||
      r.sentiment.toLowerCase().includes(term)
    );
  }, [dataset, searchTerm]);

  const totalPages = Math.ceil(filteredReviews.length / pageSize) || 1;
  const paginatedReviews = filteredReviews.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  // Handle Unlabeled File Upload for Batch Prediction
  const handleBatchFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!isTrained) {
      setBatchError('Models are not yet trained. Please train the models before running batch predictions.');
      return;
    }

    setIsProcessingBatch(true);
    setBatchError(null);

    const result = await parseUnlabeledFile(file);
    if (!result.success) {
      setBatchError(result.error || 'Failed to parse unlabeled review file.');
      setIsProcessingBatch(false);
      return;
    }

    // Classify using latest trained model with non-blocking chunking
    const predictions: BatchReviewItem[] = [];
    const chunkSize = 50;
    for (let i = 0; i < result.reviews.length; i += chunkSize) {
      const slice = result.reviews.slice(i, i + chunkSize);
      for (const item of slice) {
        predictions.push({
          id: item.id,
          review: item.review,
          prediction: sentimentEngine.predict(item.review)
        });
      }
      if (result.reviews.length > 50) {
        await new Promise(r => setTimeout(r, 0));
      }
    }

    setBatchReviews(predictions);
    setIsProcessingBatch(false);
  };

  // Export Batch Results to CSV
  const handleDownloadBatchResults = () => {
    if (batchReviews.length === 0) return;

    const exportRows = batchReviews.map(item => ({
      Review: item.review,
      'Logistic Regression': item.prediction?.logisticRegression.sentiment || '',
      'Naive Bayes': item.prediction?.naiveBayes.sentiment || '',
      'Linear SVM': item.prediction?.linearSVM.sentiment || '',
      'Random Forest': item.prediction?.randomForest.sentiment || '',
      'SGD Classifier': item.prediction?.sgdClassifier.sentiment || '',
      'Ensemble Prediction': item.prediction?.ensemble.sentiment || '',
      'Confidence (%)': item.prediction?.ensemble.confidence || ''
    }));

    downloadCSV('imdb_ensemble_predictions.csv', exportRows);
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Top Header */}
      <div
        className={`border-b pb-5 flex flex-col md:flex-row md:items-center justify-between gap-4 ${
          isDark ? 'border-slate-800' : 'border-slate-200'
        }`}
      >
        <div>
          <div className="flex items-center gap-3">
            <h1
              className={`text-xl font-bold tracking-tight ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              Dataset Management
            </h1>
            <span className="font-mono text-xs font-bold text-indigo-500 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded">
              Model v{stats.modelVersion}
            </span>
          </div>
          <p className={`text-xs mt-1 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
            Upload labeled training datasets to accumulate knowledge persistently across sessions, or classify unlabeled reviews in batch.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={onResetDataset}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              isDark
                ? 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700'
                : 'bg-slate-100 hover:bg-slate-200 text-slate-700 border-slate-300'
            }`}
            title="Reset dataset back to benchmark state"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset to Benchmark</span>
          </button>
        </div>
      </div>

      {/* Structured Training Dataset Summary as Requested */}
      <div
        className={`border rounded-xl p-5 transition-colors ${
          isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
        }`}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b pb-4 mb-4 border-inherit">
          <div>
            <h2
              className={`text-sm font-bold tracking-tight ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              Training Dataset
            </h2>
            <p className={`text-xs mt-0.5 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Cumulative storage (IndexedDB persisted)
            </p>
          </div>

          {/* Action Button: Train / Update Models */}
          <div className="flex items-center gap-3">
            {isTraining && (
              <span className="text-xs font-mono font-medium text-emerald-500 animate-pulse flex items-center gap-1.5">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>{trainingStep}</span>
              </span>
            )}
            <button
              onClick={onTrainModels}
              disabled={isTraining || dataset.length === 0}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white transition-colors shadow-sm cursor-pointer disabled:cursor-not-allowed"
            >
              {isTraining ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Training models...</span>
                </>
              ) : (
                <>
                  <Cpu className="w-3.5 h-3.5" />
                  <span>Train & Update Models</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* 6 Key Properties Displayed */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
          <div
            className={`p-3 rounded-lg border ${
              isDark ? 'bg-slate-950/50 border-slate-800' : 'bg-slate-50 border-slate-200'
            }`}
          >
            <span className={`text-[11px] font-medium block ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Total Reviews
            </span>
            <span
              className={`text-lg font-bold font-mono tabular-nums block mt-1 ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              {stats.total.toLocaleString()}
            </span>
          </div>

          <div
            className={`p-3 rounded-lg border ${
              isDark ? 'bg-slate-950/50 border-slate-800' : 'bg-slate-50 border-slate-200'
            }`}
          >
            <span className={`text-[11px] font-medium block ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Positive
            </span>
            <span className="text-lg font-bold font-mono tabular-nums text-emerald-500 block mt-1">
              {stats.positive.toLocaleString()}
            </span>
          </div>

          <div
            className={`p-3 rounded-lg border ${
              isDark ? 'bg-slate-950/50 border-slate-800' : 'bg-slate-50 border-slate-200'
            }`}
          >
            <span className={`text-[11px] font-medium block ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Negative
            </span>
            <span className="text-lg font-bold font-mono tabular-nums text-rose-500 block mt-1">
              {stats.negative.toLocaleString()}
            </span>
          </div>

          <div
            className={`p-3 rounded-lg border ${
              isDark ? 'bg-slate-950/50 border-slate-800' : 'bg-slate-50 border-slate-200'
            }`}
          >
            <span className={`text-[11px] font-medium block ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Datasets
            </span>
            <span
              className={`text-lg font-bold font-mono tabular-nums block mt-1 ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              {stats.datasetsCount}
            </span>
          </div>

          <div
            className={`p-3 rounded-lg border ${
              isDark ? 'bg-slate-950/50 border-slate-800' : 'bg-slate-50 border-slate-200'
            }`}
          >
            <span className={`text-[11px] font-medium block ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Model Version
            </span>
            <span className="text-lg font-bold font-mono text-indigo-500 block mt-1">
              Model v{stats.modelVersion}
            </span>
          </div>

          <div
            className={`p-3 rounded-lg border ${
              isDark ? 'bg-slate-950/50 border-slate-800' : 'bg-slate-50 border-slate-200'
            }`}
          >
            <span className={`text-[11px] font-medium block ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Last Trained
            </span>
            <span
              className={`text-xs font-mono truncate block mt-1 ${
                isDark ? 'text-slate-300' : 'text-slate-700'
              }`}
              title={stats.lastTrainedTime || 'Not trained yet'}
            >
              {stats.lastTrainedTime || 'Initial'}
            </span>
          </div>
        </div>
      </div>

      {/* Upload Labeled Training Dataset Box */}
      <div
        className={`border rounded-xl p-6 transition-colors ${
          isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
        }`}
      >
        <h2
          className={`text-sm font-semibold tracking-tight mb-1 ${
            isDark ? 'text-white' : 'text-slate-900'
          }`}
        >
          Upload Training Dataset (.csv, .xlsx)
        </h2>
        <p className={`text-xs mb-4 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
          File must contain columns: <span className="font-mono font-medium">review</span> and{' '}
          <span className="font-mono font-medium">sentiment</span> (e.g. positive / negative).
          New data will be appended to the accumulated training dataset.
        </p>

        <label
          className={`flex items-center justify-center gap-3 px-4 py-6 border-2 border-dashed rounded-xl cursor-pointer transition-colors ${
            isDark
              ? 'border-slate-700 hover:border-indigo-500 bg-slate-950/40'
              : 'border-slate-300 hover:border-indigo-500 bg-slate-50/70'
          }`}
        >
          <Upload className="w-5 h-5 text-indigo-500" />
          <div className="text-center sm:text-left">
            <span
              className={`text-xs font-semibold block ${
                isDark ? 'text-slate-200' : 'text-slate-800'
              }`}
            >
              {isUploading ? 'Validating and accumulating...' : 'Upload Training Dataset'}
            </span>
            <span className={`text-[11px] block mt-0.5 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
              CSV or Excel format (.csv, .xlsx, .xls)
            </span>
          </div>
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleDatasetFileUpload}
            disabled={isUploading}
            className="hidden"
          />
        </label>

        {uploadError && (
          <div className="mt-3 p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-500 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{uploadError}</span>
          </div>
        )}

        {/* Post-Upload Statistics Summary Card as Requested */}
        {lastUploadInfo && (
          <div
            className={`mt-4 p-4 rounded-xl border text-xs space-y-2 ${
              isDark
                ? 'bg-slate-950/60 border-slate-800 text-slate-300'
                : 'bg-indigo-50/50 border-indigo-200 text-slate-800'
            }`}
          >
            <div className="flex items-center justify-between font-semibold border-b pb-2 border-inherit">
              <span className="flex items-center gap-1.5 text-emerald-500">
                <CheckCircle2 className="w-4 h-4" />
                <span>Uploaded: {lastUploadInfo.filename}</span>
              </span>
              <span className="text-[11px] font-mono text-indigo-500">
                {isTraining ? 'Training models automatically...' : isTrained ? '✓ Model trained successfully' : 'Ready'}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-1 text-[11px] font-mono">
              <div>
                <span className="text-slate-500 block">Uploaded Rows:</span>
                <span className="font-bold text-indigo-500">{lastUploadInfo.uploadedRows.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-slate-500 block">New Valid Rows:</span>
                <span className="font-bold text-emerald-500">+{lastUploadInfo.newRecords.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Duplicates Removed:</span>
                <span className="font-bold text-amber-500">{lastUploadInfo.duplicatesRemoved.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Accumulated Total:</span>
                <span className="font-bold text-slate-900 dark:text-white">{stats.total.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Datasets Total:</span>
                <span className="font-bold text-indigo-500">{stats.datasetsCount}</span>
              </div>
            </div>

            <div className="pt-2 flex items-center justify-between">
              <span className="text-[11px] text-slate-500">
                {isTraining 
                  ? 'Automatically training models on all accumulated examples...' 
                  : 'Models trained automatically with all accumulated examples. Click to re-run if needed.'}
              </span>
              <button
                onClick={onTrainModels}
                disabled={isTraining || dataset.length === 0}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white transition-colors cursor-pointer disabled:cursor-not-allowed"
              >
                {isTraining ? 'Training in progress...' : 'Train & Update Models'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Dataset Preview Table */}
      <div
        className={`border rounded-xl overflow-hidden transition-colors ${
          isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
        }`}
      >
        <div
          className={`p-4 border-b flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
            isDark ? 'border-slate-800' : 'border-slate-200'
          }`}
        >
          <div className="flex items-center gap-3">
            <h2
              className={`text-sm font-semibold tracking-tight ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              Accumulated Dataset Preview
            </h2>
            <span className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Showing {filteredReviews.length} records
            </span>
          </div>

          <div className="relative">
            <Search className={`w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 ${isDark ? 'text-slate-500' : 'text-slate-400'}`} />
            <input
              type="text"
              value={searchTerm}
              onChange={e => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              placeholder="Search reviews or labels..."
              className={`border rounded-lg pl-8 pr-3 py-1.5 text-xs focus:outline-none focus:border-indigo-500 w-full sm:w-64 transition-colors ${
                isDark
                  ? 'bg-slate-950 border-slate-700 text-white placeholder-slate-500'
                  : 'bg-slate-50 border-slate-300 text-slate-900 placeholder-slate-400'
              }`}
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead
              className={`font-medium border-b ${
                isDark
                  ? 'bg-slate-950/60 text-slate-400 border-slate-800'
                  : 'bg-slate-50 text-slate-500 border-slate-200'
              }`}
            >
              <tr>
                <th className="py-2.5 px-4 w-16">#</th>
                <th className="py-2.5 px-4">Review Text</th>
                <th className="py-2.5 px-4 w-32">Sentiment Label</th>
              </tr>
            </thead>
            <tbody className={`divide-y ${isDark ? 'divide-slate-800/60' : 'divide-slate-200'}`}>
              {paginatedReviews.map((r, index) => {
                const globalIndex = (currentPage - 1) * pageSize + index + 1;
                return (
                  <tr
                    key={r.id}
                    className={`transition-colors ${
                      isDark ? 'hover:bg-slate-800/40 text-slate-200' : 'hover:bg-slate-50 text-slate-800'
                    }`}
                  >
                    <td className="py-2.5 px-4 font-mono text-slate-400 tabular-nums">
                      {globalIndex}
                    </td>
                    <td className="py-2.5 px-4 max-w-xl">
                      <p className="line-clamp-2 leading-relaxed">{r.review}</p>
                    </td>
                    <td className="py-2.5 px-4">
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium capitalize ${
                          r.sentiment === 'positive'
                            ? 'text-emerald-500 bg-emerald-500/10'
                            : 'text-rose-500 bg-rose-500/10'
                        }`}
                      >
                        {r.sentiment}
                      </span>
                    </td>
                  </tr>
                );
              })}
              {paginatedReviews.length === 0 && (
                <tr>
                  <td colSpan={3} className="py-8 text-center text-slate-500 text-xs">
                    No matching reviews found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination controls */}
        {totalPages > 1 && (
          <div
            className={`px-4 py-3 border-t flex items-center justify-between text-xs ${
              isDark
                ? 'border-slate-800 text-slate-400'
                : 'border-slate-200 text-slate-600'
            }`}
          >
            <span>
              Page {currentPage} of {totalPages}
            </span>
            <div className="flex items-center gap-1">
              <button
                disabled={currentPage === 1}
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                className={`px-2.5 py-1 rounded disabled:opacity-30 disabled:cursor-not-allowed ${
                  isDark
                    ? 'bg-slate-800 hover:bg-slate-700 text-white'
                    : 'bg-slate-200 hover:bg-slate-300 text-slate-800'
                }`}
              >
                Previous
              </button>
              <button
                disabled={currentPage === totalPages}
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                className={`px-2.5 py-1 rounded disabled:opacity-30 disabled:cursor-not-allowed ${
                  isDark
                    ? 'bg-slate-800 hover:bg-slate-700 text-white'
                    : 'bg-slate-200 hover:bg-slate-300 text-slate-800'
                }`}
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Batch Unlabeled Review Prediction Section */}
      <div
        className={`border rounded-xl p-6 space-y-4 transition-colors ${
          isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
        }`}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <h2
                className={`text-sm font-semibold tracking-tight ${
                  isDark ? 'text-white' : 'text-slate-900'
                }`}
              >
                Batch Review Prediction & CSV Export
              </h2>
              <span className="font-mono text-[10px] text-indigo-500 bg-indigo-500/10 px-1.5 py-0.5 rounded">
                Model v{stats.modelVersion}
              </span>
            </div>
            <p className={`text-xs mt-0.5 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Upload an unlabeled file (.csv or .xlsx with a <span className="font-mono font-medium">review</span> column) to classify using the latest trained model.
            </p>
          </div>

          {batchReviews.length > 0 && (
            <button
              onClick={handleDownloadBatchResults}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition-colors shadow-sm"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download Results (CSV)</span>
            </button>
          )}
        </div>

        <label
          className={`flex items-center justify-center gap-3 px-4 py-5 border-2 border-dashed rounded-xl cursor-pointer transition-colors ${
            isDark
              ? 'border-slate-700 hover:border-emerald-500 bg-slate-950/40'
              : 'border-slate-300 hover:border-emerald-500 bg-slate-50/70'
          }`}
        >
          <Upload className="w-4 h-4 text-emerald-500" />
          <span
            className={`text-xs font-medium ${
              isDark ? 'text-slate-300' : 'text-slate-700'
            }`}
          >
            {isProcessingBatch
              ? `Classifying reviews with Model v${stats.modelVersion}...`
              : 'Upload Unlabeled Reviews (.csv or .xlsx)'}
          </span>
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleBatchFileUpload}
            disabled={isProcessingBatch}
            className="hidden"
          />
        </label>

        {batchError && (
          <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-500 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{batchError}</span>
          </div>
        )}

        {/* Batch Results Table */}
        {batchReviews.length > 0 && (
          <div
            className={`border rounded-lg overflow-hidden mt-4 ${
              isDark ? 'border-slate-800' : 'border-slate-200'
            }`}
          >
            <div
              className={`px-4 py-2.5 border-b text-xs flex items-center justify-between ${
                isDark
                  ? 'bg-slate-950 border-slate-800 text-slate-400'
                  : 'bg-slate-50 border-slate-200 text-slate-600'
              }`}
            >
              <span>{batchReviews.length} Batch Predictions Completed (Model v{stats.modelVersion})</span>
              <span className="text-emerald-500 font-mono text-[11px] font-medium">
                Ready for CSV export
              </span>
            </div>
            <div className="overflow-x-auto max-h-80">
              <table className="w-full text-left text-xs">
                <thead
                  className={`font-medium sticky top-0 border-b ${
                    isDark
                      ? 'bg-slate-950/80 text-slate-400 border-slate-800'
                      : 'bg-slate-100 text-slate-600 border-slate-200'
                  }`}
                >
                  <tr>
                    <th className="py-2 px-3">Review</th>
                    <th className="py-2 px-3 text-center">LR</th>
                    <th className="py-2 px-3 text-center">NB</th>
                    <th className="py-2 px-3 text-center">SVM</th>
                    <th className="py-2 px-3 text-center">RF</th>
                    <th className="py-2 px-3 text-center">SGD</th>
                    <th className="py-2 px-3 text-right">Ensemble Sentiment</th>
                    <th className="py-2 px-3 text-right">Confidence</th>
                  </tr>
                </thead>
                <tbody
                  className={`divide-y font-mono text-[11px] ${
                    isDark ? 'divide-slate-800/60' : 'divide-slate-200'
                  }`}
                >
                  {batchReviews.slice(0, 50).map((b, i) => (
                    <tr
                      key={i}
                      className={isDark ? 'hover:bg-slate-800/30' : 'hover:bg-slate-50'}
                    >
                      <td
                        className={`py-2 px-3 font-sans max-w-sm truncate ${
                          isDark ? 'text-slate-200' : 'text-slate-800'
                        }`}
                      >
                        {b.review}
                      </td>
                      <td className="py-2 px-3 text-center">
                        <span className={b.prediction?.logisticRegression.sentiment === 'positive' ? 'text-emerald-500' : 'text-rose-500'}>
                          {b.prediction?.logisticRegression.sentiment === 'positive' ? 'POS' : 'NEG'}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-center">
                        <span className={b.prediction?.naiveBayes.sentiment === 'positive' ? 'text-emerald-500' : 'text-rose-500'}>
                          {b.prediction?.naiveBayes.sentiment === 'positive' ? 'POS' : 'NEG'}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-center">
                        <span className={b.prediction?.linearSVM.sentiment === 'positive' ? 'text-emerald-500' : 'text-rose-500'}>
                          {b.prediction?.linearSVM.sentiment === 'positive' ? 'POS' : 'NEG'}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-center">
                        <span className={b.prediction?.randomForest.sentiment === 'positive' ? 'text-emerald-500' : 'text-rose-500'}>
                          {b.prediction?.randomForest.sentiment === 'positive' ? 'POS' : 'NEG'}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-center">
                        <span className={b.prediction?.sgdClassifier.sentiment === 'positive' ? 'text-emerald-500' : 'text-rose-500'}>
                          {b.prediction?.sgdClassifier.sentiment === 'positive' ? 'POS' : 'NEG'}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-right font-sans font-semibold capitalize">
                        <span className={b.prediction?.ensemble.sentiment === 'positive' ? 'text-emerald-500' : 'text-rose-500'}>
                          {b.prediction?.ensemble.sentiment}
                        </span>
                      </td>
                      <td
                        className={`py-2 px-3 text-right ${
                          isDark ? 'text-slate-300' : 'text-slate-700'
                        }`}
                      >
                        {b.prediction?.ensemble.confidence}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {batchReviews.length > 50 && (
              <div
                className={`p-2 text-center text-[11px] border-t ${
                  isDark
                    ? 'text-slate-500 bg-slate-950 border-slate-800'
                    : 'text-slate-500 bg-slate-50 border-slate-200'
                }`}
              >
                Displaying first 50 results. Click "Download Results" to get the full CSV file.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
