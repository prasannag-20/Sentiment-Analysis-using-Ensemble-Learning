import React from 'react';
import { 
  Cpu, 
  CheckCircle2, 
  Loader2, 
  ArrowRight,
  Database,
  FileSpreadsheet,
  Clock
} from 'lucide-react';
import { ActiveTab, DatasetStats, EvaluationResults, ReviewItem, Theme } from '../../types';

interface TrainSectionProps {
  dataset: ReviewItem[];
  stats: DatasetStats;
  isTrained: boolean;
  evaluation: EvaluationResults | null;
  isTraining: boolean;
  trainingStep: string;
  trainingProgress: number;
  onTrainModels: () => Promise<EvaluationResults | void>;
  onNavigate: (tab: ActiveTab) => void;
  theme: Theme;
}

export const TrainSection: React.FC<TrainSectionProps> = ({
  dataset,
  stats,
  isTrained,
  evaluation,
  isTraining,
  trainingStep,
  trainingProgress,
  onTrainModels,
  onNavigate,
  theme
}) => {
  const isDark = theme === 'dark';

  const modelStatusItems = [
    { name: 'Logistic Regression', description: 'Log-loss optimization with L2 regularization' },
    { name: 'Multinomial Naive Bayes', description: 'Laplace-smoothed unigram/bigram log-likelihood' },
    { name: 'Linear SVM', description: 'Hinge loss SGD with Platt probability calibration' },
    { name: 'Random Forest', description: 'Bagged decision trees with feature subsampling' },
    { name: 'SGD Classifier', description: 'Modified Huber loss linear stochastic gradient descent' },
    { name: 'Soft Voting Ensemble', description: 'Probability-weighted consensus aggregation' }
  ];

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
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
              Model Training
            </h1>
            <span className="font-mono text-xs font-bold text-indigo-500 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded">
              Model v{stats.modelVersion}
            </span>
          </div>
          <p className={`text-xs mt-1 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
            Train all 5 classifiers and the Soft Voting Ensemble cumulatively using all {stats.total.toLocaleString()} accumulated training examples.
          </p>
        </div>

        <button
          onClick={onTrainModels}
          disabled={isTraining || dataset.length === 0}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors shadow-md shadow-indigo-600/20 whitespace-nowrap self-start md:self-auto cursor-pointer disabled:cursor-not-allowed"
        >
          {isTraining ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-white" />
              <span>Training in progress ({trainingProgress}%)...</span>
            </>
          ) : (
            <>
              <Cpu className="w-4 h-4" />
              <span>{isTrained ? 'Train & Update Models' : 'Train Models'}</span>
            </>
          )}
        </button>
      </div>

      {/* Dataset & Configuration Info Cards - All values dynamically calculated */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Card 1: Accumulated Training Examples */}
        <div
          className={`border rounded-xl p-5 transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <span className={`text-[11px] font-medium ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
            Accumulated Training Examples
          </span>
          <div
            className={`mt-2 text-2xl font-bold font-mono tabular-nums ${
              isDark ? 'text-white' : 'text-slate-900'
            }`}
          >
            {stats.total.toLocaleString()}
          </div>
          <div className={`mt-1 text-xs tabular-nums ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
            Pos: <span className="text-emerald-500 font-semibold">{stats.positive.toLocaleString()}</span> · Neg: <span className="text-rose-500 font-semibold">{stats.negative.toLocaleString()}</span>
          </div>
        </div>

        {/* Card 2: Datasets Uploaded */}
        <div
          className={`border rounded-xl p-5 transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <span className={`text-[11px] font-medium ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
            Datasets Uploaded
          </span>
          <div className="mt-2 text-2xl font-bold text-indigo-500 font-mono tabular-nums">
            {stats.datasetsCount}
          </div>
          <div className={`mt-1 text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            {stats.datasetsCount === 1 ? '1 dataset contributing' : `${stats.datasetsCount} datasets accumulated`}
          </div>
        </div>

        {/* Card 3: Training / Testing Split */}
        <div
          className={`border rounded-xl p-5 transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <span className={`text-[11px] font-medium ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
            Train (80%) / Test (20%)
          </span>
          <div className="mt-2 text-xl font-bold font-mono tabular-nums">
            <span className="text-indigo-500">{stats.trainCount.toLocaleString()}</span>
            <span className={isDark ? 'text-slate-600 mx-1' : 'text-slate-300 mx-1'}>/</span>
            <span className="text-sky-500">{stats.testCount.toLocaleString()}</span>
          </div>
          <div className={`mt-1 text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            Stratified train/test split
          </div>
        </div>

        {/* Card 4: Model Version (Clean, Simplified without timestamp) */}
        <div
          className={`border rounded-xl p-5 transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <span className={`text-[11px] font-medium ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
            Model Version
          </span>
          <div className="mt-2 text-2xl font-bold font-mono text-indigo-500">
            Model v{stats.modelVersion}
          </div>
          <div className={`mt-1 text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
            {isTrained ? (
              <span className="text-emerald-500 font-medium">Active & Trained</span>
            ) : (
              <span className="text-amber-500 font-medium">Pending Retraining</span>
            )}
          </div>
        </div>
      </div>

      {/* Uploaded Training Datasets (Clean & Compact) */}
      <div
        className={`border rounded-xl p-5 space-y-3 transition-colors ${
          isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
        }`}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b pb-3 border-inherit">
          <div>
            <h2 className={`text-sm font-bold tracking-tight ${isDark ? 'text-white' : 'text-slate-900'}`}>
              Uploaded Training Datasets
            </h2>
            <p className={`text-xs mt-0.5 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Review the individual datasets accumulated into the current training pipeline
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded bg-indigo-500/10 text-indigo-500 border border-indigo-500/20">
              Accumulated Total: {stats.total.toLocaleString()} examples
            </span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className={`border-b text-[11px] font-semibold ${isDark ? 'border-slate-800 text-slate-400 bg-slate-950/40' : 'border-slate-200 text-slate-600 bg-slate-50'}`}>
                <th className="py-2.5 px-4">Dataset Name</th>
                <th className="py-2.5 px-4 text-right">Number of Examples</th>
                <th className="py-2.5 px-4 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-inherit font-mono">
              {stats.uploadedDatasets && stats.uploadedDatasets.length > 0 ? (
                stats.uploadedDatasets.map((ds, idx) => (
                  <tr key={ds.id || idx} className={isDark ? 'hover:bg-slate-800/30' : 'hover:bg-slate-50'}>
                    <td className="py-3 px-4 font-sans flex items-center gap-2 font-medium">
                      <FileSpreadsheet className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
                      <span className={isDark ? 'text-slate-200' : 'text-slate-800'}>{ds.name}</span>
                    </td>
                    <td className="py-3 px-4 text-right tabular-nums font-semibold">
                      {ds.examplesCount.toLocaleString()} examples
                    </td>
                    <td className="py-3 px-4 text-right font-sans">
                      {ds.status === 'Trained' ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                          <CheckCircle2 className="w-3 h-3" />
                          <span>Trained</span>
                        </span>
                      ) : ds.status === 'Training' ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-indigo-500/10 text-indigo-500 border border-indigo-500/20 animate-pulse">
                          <Loader2 className="w-3 h-3 animate-spin" />
                          <span>Training...</span>
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-500/10 text-amber-500 border border-amber-500/20">
                          <Clock className="w-3 h-3" />
                          <span>Pending Training</span>
                        </span>
                      )}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={3} className="py-4 text-center text-slate-500 font-sans">
                    No datasets uploaded yet.
                  </td>
                </tr>
              )}
            </tbody>
            <tfoot>
              <tr className={`border-t font-semibold ${isDark ? 'border-slate-800 bg-slate-950/60 text-slate-200' : 'border-slate-200 bg-slate-50 text-slate-800'}`}>
                <td className="py-3 px-4 font-sans">
                  Total Accumulated Training Examples
                </td>
                <td className="py-3 px-4 text-right font-mono text-indigo-500 font-bold text-sm">
                  {stats.total.toLocaleString()} examples
                </td>
                <td className="py-3 px-4 text-right font-sans">
                  {isTrained ? (
                    <span className="text-emerald-500 text-xs font-semibold">All Models Trained ✓</span>
                  ) : (
                    <span className="text-amber-500 text-xs font-semibold">Ready for Training</span>
                  )}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>

      {/* Training Progress / Live Status Box */}
      <div
        className={`border rounded-xl p-6 space-y-4 transition-colors ${
          isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
        }`}
      >
        <div className="flex items-center justify-between">
          <div>
            <h2
              className={`text-sm font-semibold tracking-tight ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              Training Status
            </h2>
            <p className="text-xs text-indigo-500 font-mono font-medium mt-0.5">
              {trainingStep}
            </p>
          </div>
          <span className="text-xs font-mono font-semibold text-indigo-500 tabular-nums">
            {isTraining
              ? `${trainingProgress}%`
              : isTrained
              ? 'Model trained successfully! (100%)'
              : 'Ready'}
          </span>
        </div>

        {/* Progress Bar */}
        <div
          className={`w-full h-2.5 rounded-full overflow-hidden border ${
            isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-200'
          }`}
        >
          <div
            className="h-full bg-gradient-to-r from-indigo-500 via-sky-400 to-emerald-500 transition-all duration-300 rounded-full"
            style={{ width: `${isTrained && !isTraining ? 100 : trainingProgress}%` }}
          />
        </div>

        {/* Model Status Checklist */}
        <div className="pt-2">
          <div className={`text-xs font-semibold mb-3 ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
            Model Verification Status (Model v{stats.modelVersion}):
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {modelStatusItems.map((item, index) => {
              const completed = isTrained && !isTraining;
              return (
                <div
                  key={index}
                  className={`p-3 rounded-lg border transition-colors flex items-center justify-between ${
                    completed
                      ? isDark
                        ? 'bg-slate-950/60 border-slate-800'
                        : 'bg-slate-50 border-slate-200'
                      : isDark
                      ? 'bg-slate-950/30 border-slate-800 opacity-60'
                      : 'bg-slate-50/40 border-slate-200 opacity-60'
                  }`}
                >
                  <div>
                    <div
                      className={`text-xs font-semibold ${
                        isDark ? 'text-slate-200' : 'text-slate-800'
                      }`}
                    >
                      {item.name}
                    </div>
                    <div className={`text-[11px] mt-0.5 ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>
                      {item.description}
                    </div>
                  </div>
                  <div className="shrink-0 ml-3">
                    {completed ? (
                      <div className="flex items-center gap-1 text-emerald-500 text-xs font-medium">
                        <CheckCircle2 className="w-4 h-4" />
                        <span>Ready ✓</span>
                      </div>
                    ) : isTraining ? (
                      <Loader2 className="w-4 h-4 text-indigo-500 animate-spin" />
                    ) : (
                      <span className="text-[11px] text-slate-400 font-mono">Pending</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Post-Training Call to Action */}
        {isTrained && !isTraining && (
          <div
            className={`pt-4 border-t flex flex-col sm:flex-row items-center justify-between gap-3 ${
              isDark ? 'border-slate-800' : 'border-slate-200'
            }`}
          >
            <div className="text-xs text-emerald-500 flex items-center gap-1.5 font-medium">
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              <span>
                Model v{stats.modelVersion} trained and saved persistently. Ready for predictions.
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => onNavigate('evaluation')}
                className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold border transition-colors cursor-pointer ${
                  isDark
                    ? 'bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700'
                    : 'bg-slate-100 hover:bg-slate-200 text-slate-800 border-slate-300'
                }`}
              >
                <span>View Evaluation</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => onNavigate('chatbot')}
                className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition-colors shadow-sm cursor-pointer"
              >
                <span>Test in Chatbot</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
