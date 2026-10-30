import React from 'react';
import { 
  Database, 
  ThumbsUp, 
  ThumbsDown, 
  Cpu, 
  Target, 
  Award,
  ArrowRight,
  Bot,
  FileCheck,
  CheckCircle2,
  Clock,
  Layers
} from 'lucide-react';
import { ActiveTab, DatasetStats, EvaluationResults, Theme } from '../../types';

interface DashboardSectionProps {
  stats: DatasetStats;
  evaluation: EvaluationResults | null;
  isTrained: boolean;
  theme: Theme;
  onNavigate: (tab: ActiveTab) => void;
}

export const DashboardSection: React.FC<DashboardSectionProps> = ({
  stats,
  evaluation,
  isTrained,
  theme,
  onNavigate
}) => {
  const isDark = theme === 'dark';
  const bestAcc = evaluation ? evaluation.bestAccuracy : 0;
  const bestModel = evaluation ? evaluation.bestModelName : 'Soft Voting Ensemble';

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Hero Header */}
      <div
        className={`border-b pb-6 flex flex-col md:flex-row md:items-center justify-between gap-4 ${
          isDark ? 'border-slate-800' : 'border-slate-200'
        }`}
      >
        <div>
          <div className="flex items-center gap-3">
            <h1
              className={`text-2xl font-bold tracking-tight ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              Sentiment Analysis Using Ensemble Learning
            </h1>
            <span className="font-mono text-xs font-bold text-indigo-500 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded">
              Model v{stats.modelVersion}
            </span>
          </div>
          <p className={`text-sm mt-1 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
            Cumulative machine-learning sentiment classification pipeline for IMDb movie reviews using 5 base classifiers and Soft Voting.
          </p>
        </div>

        {stats.isDemo ? (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-500">
            <span>Demo Benchmark Dataset Loaded</span>
            <button
              onClick={() => onNavigate('dataset')}
              className="text-amber-600 dark:text-amber-400 underline font-medium hover:opacity-80 ml-1"
            >
              Upload Custom
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-500">
            <span>{stats.datasetsCount} Datasets Accumulated</span>
            <span className="text-slate-400">·</span>
            <span className="font-mono">{stats.total} total reviews</span>
          </div>
        )}
      </div>

      {/* KPI Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Total Reviews */}
        <div
          className={`border rounded-xl p-5 transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className={`text-xs font-medium ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Total Reviews
            </span>
            <Database className="w-4 h-4 text-indigo-500" />
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span
              className={`text-2xl font-bold tabular-nums ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              {stats.total.toLocaleString()}
            </span>
            <span className={`text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
              accumulated
            </span>
          </div>
          <div className={`mt-2 text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            From {stats.datasetsCount} uploaded {stats.datasetsCount === 1 ? 'batch' : 'batches'}
          </div>
        </div>

        {/* Sentiment Distribution */}
        <div
          className={`border rounded-xl p-5 transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className={`text-xs font-medium ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Positive / Negative
            </span>
            <div className="flex items-center gap-1">
              <ThumbsUp className="w-3.5 h-3.5 text-emerald-500" />
              <ThumbsDown className="w-3.5 h-3.5 text-rose-500" />
            </div>
          </div>
          <div className="mt-3 flex items-baseline gap-3">
            <div>
              <span className="text-xl font-bold text-emerald-500 tabular-nums">
                {stats.positive}
              </span>
              <span className={`text-[11px] ml-1 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                Pos
              </span>
            </div>
            <span className={isDark ? 'text-slate-700' : 'text-slate-300'}>/</span>
            <div>
              <span className="text-xl font-bold text-rose-500 tabular-nums">
                {stats.negative}
              </span>
              <span className={`text-[11px] ml-1 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                Neg
              </span>
            </div>
          </div>
          <div className={`mt-2 text-[11px] tabular-nums ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            {stats.total > 0 ? `${((stats.positive / stats.total) * 100).toFixed(0)}% positive ratio` : '0%'}
          </div>
        </div>

        {/* Train / Test Split Sizes */}
        <div
          className={`border rounded-xl p-5 transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className={`text-xs font-medium ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Train / Test Partition
            </span>
            <Target className="w-4 h-4 text-indigo-500" />
          </div>
          <div className="mt-3 flex items-baseline gap-3">
            <div>
              <span className="text-xl font-bold text-indigo-500 tabular-nums">
                {stats.trainCount}
              </span>
              <span className={`text-[11px] ml-1 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                Train (80%)
              </span>
            </div>
            <span className={isDark ? 'text-slate-700' : 'text-slate-300'}>/</span>
            <div>
              <span className="text-xl font-bold text-sky-500 tabular-nums">
                {stats.testCount}
              </span>
              <span className={`text-[11px] ml-1 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                Test (20%)
              </span>
            </div>
          </div>
          <div className={`mt-2 text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
            Stratified 80/20 partition
          </div>
        </div>

        {/* Best Model Accuracy & Model Version */}
        <div
          className={`border rounded-xl p-5 transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className={`text-xs font-medium ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Best Model Accuracy
            </span>
            <Award className="w-4 h-4 text-amber-500" />
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span
              className={`text-2xl font-bold tabular-nums ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              {bestAcc > 0 ? `${bestAcc}%` : 'Pending'}
            </span>
            <span className="text-xs text-indigo-500 font-mono font-semibold">
              v{stats.modelVersion}
            </span>
          </div>
          <div className={`mt-2 text-[11px] truncate ${isDark ? 'text-slate-400' : 'text-slate-600'}`} title={bestModel}>
            {bestModel}
          </div>
        </div>
      </div>

      {/* Model Versioning & Pipeline Status Bar (Clean, without timestamp) */}
      <div
        className={`border rounded-xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs ${
          isDark
            ? 'bg-slate-900/60 border-slate-800 text-slate-300'
            : 'bg-white border-slate-200 text-slate-700 shadow-sm'
        }`}
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-500">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <span className="font-semibold text-slate-900 dark:text-white">Active Model: </span>
            <span className="font-mono text-indigo-500 font-bold">Model v{stats.modelVersion}</span>
            <span className="mx-2 text-slate-400">·</span>
            <span>Accumulated dataset: {stats.total.toLocaleString()} unique reviews ({stats.datasetsCount} {stats.datasetsCount === 1 ? 'dataset' : 'datasets'})</span>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-500">Status:</span>
          {isTrained ? (
            <span className="inline-flex items-center gap-1 text-emerald-500 font-medium">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Trained & Ready</span>
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-amber-500 font-medium">
              <Clock className="w-3.5 h-3.5" />
              <span>Pending Retraining</span>
            </span>
          )}
        </div>
      </div>

      {/* Workflow Component */}
      <div
        className={`border rounded-xl p-6 transition-colors ${
          isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
        }`}
      >
        <div className="flex items-center justify-between mb-4">
          <h2
            className={`text-sm font-semibold tracking-tight ${
              isDark ? 'text-white' : 'text-slate-900'
            }`}
          >
            Cumulative ML Pipeline Workflow
          </h2>
          <span className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
            Dataset → Train → Predict → Evaluate
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 relative">
          {/* Step 1: Dataset */}
          <div
            onClick={() => onNavigate('dataset')}
            className={`cursor-pointer group p-4 rounded-lg border transition-all ${
              isDark
                ? 'bg-slate-950/60 border-slate-800 hover:border-indigo-500/50'
                : 'bg-slate-50 border-slate-200 hover:border-indigo-500 hover:bg-white'
            }`}
          >
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="font-semibold text-indigo-500">Step 1</span>
              <Database className="w-4 h-4 text-slate-400 group-hover:text-indigo-500 transition-colors" />
            </div>
            <div
              className={`font-semibold text-sm group-hover:text-indigo-500 ${
                isDark ? 'text-slate-200' : 'text-slate-800'
              }`}
            >
              Dataset
            </div>
            <div className={`text-xs mt-1 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Accumulate reviews (CSV/Excel) & batch input
            </div>
          </div>

          {/* Step 2: Train */}
          <div
            onClick={() => onNavigate('train')}
            className={`cursor-pointer group p-4 rounded-lg border transition-all ${
              isDark
                ? 'bg-slate-950/60 border-slate-800 hover:border-indigo-500/50'
                : 'bg-slate-50 border-slate-200 hover:border-indigo-500 hover:bg-white'
            }`}
          >
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="font-semibold text-indigo-500">Step 2</span>
              <Cpu className="w-4 h-4 text-slate-400 group-hover:text-indigo-500 transition-colors" />
            </div>
            <div
              className={`font-semibold text-sm group-hover:text-indigo-500 ${
                isDark ? 'text-slate-200' : 'text-slate-800'
              }`}
            >
              Train
            </div>
            <div className={`text-xs mt-1 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Cumulative retraining & 5 ML models + Soft Voting
            </div>
          </div>

          {/* Step 3: Predict */}
          <div
            onClick={() => onNavigate('chatbot')}
            className={`cursor-pointer group p-4 rounded-lg border transition-all ${
              isDark
                ? 'bg-slate-950/60 border-slate-800 hover:border-indigo-500/50'
                : 'bg-slate-50 border-slate-200 hover:border-indigo-500 hover:bg-white'
            }`}
          >
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="font-semibold text-indigo-500">Step 3</span>
              <Bot className="w-4 h-4 text-slate-400 group-hover:text-indigo-500 transition-colors" />
            </div>
            <div
              className={`font-semibold text-sm group-hover:text-indigo-500 ${
                isDark ? 'text-slate-200' : 'text-slate-800'
              }`}
            >
              Predict
            </div>
            <div className={`text-xs mt-1 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Sentiment chatbot & single/batch inference (Model v{stats.modelVersion})
            </div>
          </div>

          {/* Step 4: Evaluate */}
          <div
            onClick={() => onNavigate('evaluation')}
            className={`cursor-pointer group p-4 rounded-lg border transition-all ${
              isDark
                ? 'bg-slate-950/60 border-slate-800 hover:border-indigo-500/50'
                : 'bg-slate-50 border-slate-200 hover:border-indigo-500 hover:bg-white'
            }`}
          >
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="font-semibold text-indigo-500">Step 4</span>
              <FileCheck className="w-4 h-4 text-slate-400 group-hover:text-indigo-500 transition-colors" />
            </div>
            <div
              className={`font-semibold text-sm group-hover:text-indigo-500 ${
                isDark ? 'text-slate-200' : 'text-slate-800'
              }`}
            >
              Evaluate
            </div>
            <div className={`text-xs mt-1 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Test metrics, F1-scores & confusion matrix
            </div>
          </div>
        </div>
      </div>

      {/* Model Benchmark Summary Table */}
      {evaluation && (
        <div
          className={`border rounded-xl overflow-hidden transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <div
            className={`px-6 py-4 border-b flex items-center justify-between ${
              isDark ? 'border-slate-800' : 'border-slate-200'
            }`}
          >
            <div>
              <h2
                className={`text-sm font-semibold tracking-tight ${
                  isDark ? 'text-white' : 'text-slate-900'
                }`}
              >
                Model Performance Snapshot (Model v{stats.modelVersion})
              </h2>
              <p className={`text-xs mt-0.5 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                Evaluated on actual {evaluation.testSetSize} held-out test samples
              </p>
            </div>
            <button
              onClick={() => onNavigate('evaluation')}
              className="text-xs text-indigo-500 hover:text-indigo-600 flex items-center gap-1 font-semibold"
            >
              <span>View Full Evaluation</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead
                className={`font-medium border-b ${
                  isDark
                    ? 'bg-slate-950/50 text-slate-400 border-slate-800'
                    : 'bg-slate-50 text-slate-500 border-slate-200'
                }`}
              >
                <tr>
                  <th className="py-3 px-6">Model Architecture</th>
                  <th className="py-3 px-6 text-right">Accuracy</th>
                  <th className="py-3 px-6 text-right">Precision</th>
                  <th className="py-3 px-6 text-right">Recall</th>
                  <th className="py-3 px-6 text-right">F1-Score</th>
                </tr>
              </thead>
              <tbody
                className={`divide-y font-mono tabular-nums ${
                  isDark ? 'divide-slate-800/60' : 'divide-slate-200'
                }`}
              >
                {Object.values(evaluation.models).map((m, idx) => {
                  const isEnsemble = m.modelName.includes('Ensemble');
                  return (
                    <tr
                      key={idx}
                      className={
                        isEnsemble
                          ? isDark
                            ? 'bg-indigo-950/20 font-semibold text-slate-100'
                            : 'bg-indigo-50/70 font-semibold text-slate-900'
                          : isDark
                          ? 'hover:bg-slate-800/30 text-slate-300'
                          : 'hover:bg-slate-50 text-slate-700'
                      }
                    >
                      <td className="py-3 px-6 font-sans flex items-center gap-2">
                        {isEnsemble && (
                          <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 shrink-0" />
                        )}
                        <span>{m.modelName}</span>
                      </td>
                      <td className="py-3 px-6 text-right text-indigo-500 font-semibold">{m.accuracy}%</td>
                      <td className="py-3 px-6 text-right">{m.precision}%</td>
                      <td className="py-3 px-6 text-right">{m.recall}%</td>
                      <td className="py-3 px-6 text-right text-emerald-500 font-semibold">{m.f1Score}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
