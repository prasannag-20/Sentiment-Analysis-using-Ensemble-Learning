import React from 'react';
import { 
  ClipboardCheck, 
  Award, 
  ArrowRight
} from 'lucide-react';
import { ActiveTab, EvaluationResults, Theme } from '../../types';

interface EvaluationSectionProps {
  evaluation: EvaluationResults | null;
  isTrained: boolean;
  modelVersion: number;
  theme: Theme;
  onNavigate: (tab: ActiveTab) => void;
}

export const EvaluationSection: React.FC<EvaluationSectionProps> = ({
  evaluation,
  isTrained,
  modelVersion,
  theme,
  onNavigate
}) => {
  const isDark = theme === 'dark';

  if (!isTrained || !evaluation) {
    return (
      <div
        className={`max-w-4xl mx-auto py-12 text-center border rounded-xl p-8 space-y-4 transition-colors ${
          isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
        }`}
      >
        <div className="w-12 h-12 rounded-xl bg-indigo-600/20 text-indigo-500 mx-auto flex items-center justify-center">
          <ClipboardCheck className="w-6 h-6" />
        </div>
        <h2 className={`text-base font-semibold ${isDark ? 'text-white' : 'text-slate-900'}`}>
          Models Not Yet Evaluated
        </h2>
        <p className={`text-xs max-w-md mx-auto ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
          Please train the pipeline in the Train Model section to evaluate genuine test metrics on the 20% test split.
        </p>
        <button
          onClick={() => onNavigate('train')}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition-colors cursor-pointer"
        >
          <span>Go to Train Model</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
    );
  }

  const ensembleMetrics = evaluation.models.ensemble;
  const cm = ensembleMetrics.confusionMatrix;

  const modelList = [
    evaluation.models.logisticRegression,
    evaluation.models.naiveBayes,
    evaluation.models.linearSVM,
    evaluation.models.randomForest,
    evaluation.models.sgdClassifier,
    evaluation.models.ensemble
  ];

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
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
              Model Evaluation
            </h1>
            <span className="font-mono text-xs font-bold text-indigo-500 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded">
              Model v{modelVersion}
            </span>
          </div>
          <p className={`text-xs mt-1 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
            Genuine test set metrics evaluated across all 5 base classifiers and the Soft Voting Ensemble ({evaluation.testSetSize} held-out test reviews).
          </p>
        </div>

        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs ${
            isDark
              ? 'bg-indigo-950/40 border-indigo-500/30 text-slate-300'
              : 'bg-indigo-50 border-indigo-200 text-slate-800'
          }`}
        >
          <Award className="w-4 h-4 text-amber-500" />
          <span className={isDark ? 'text-slate-300' : 'text-slate-600'}>Top Performer:</span>
          <span className="font-semibold text-indigo-600 dark:text-indigo-300">
            {evaluation.bestModelName}
          </span>
          <span className="font-mono text-emerald-500 font-bold tabular-nums">
            ({evaluation.bestAccuracy}%)
          </span>
        </div>
      </div>

      {/* Main Model Metrics Comparison Table */}
      <div
        className={`border rounded-xl overflow-hidden transition-colors ${
          isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
        }`}
      >
        <div
          className={`p-4 border-b flex items-center justify-between ${
            isDark ? 'border-slate-800' : 'border-slate-200'
          }`}
        >
          <h2
            className={`text-sm font-semibold tracking-tight ${
              isDark ? 'text-white' : 'text-slate-900'
            }`}
          >
            Classifier Performance Comparison (Model v{modelVersion})
          </h2>
          <span className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
            Evaluated on held-out stratified test set
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead
              className={`font-medium border-b ${
                isDark
                  ? 'bg-slate-950/70 text-slate-400 border-slate-800'
                  : 'bg-slate-50 text-slate-500 border-slate-200'
              }`}
            >
              <tr>
                <th className="py-3 px-5">Model</th>
                <th className="py-3 px-5 text-right">Accuracy (%)</th>
                <th className="py-3 px-5 text-right">Precision (%)</th>
                <th className="py-3 px-5 text-right">Recall (%)</th>
                <th className="py-3 px-5 text-right">F1-Score (%)</th>
                <th className="py-3 px-5 text-center">Status</th>
              </tr>
            </thead>
            <tbody
              className={`divide-y font-mono tabular-nums ${
                isDark ? 'divide-slate-800/60' : 'divide-slate-200'
              }`}
            >
              {modelList.map((m, idx) => {
                const isEnsemble = m.modelName.includes('Ensemble');
                const isBest = m.modelName === evaluation.bestModelName;

                return (
                  <tr
                    key={idx}
                    className={`transition-colors ${
                      isEnsemble
                        ? isDark
                          ? 'bg-indigo-950/30 font-semibold text-white'
                          : 'bg-indigo-50/70 font-semibold text-slate-900'
                        : isDark
                        ? 'hover:bg-slate-800/40 text-slate-300'
                        : 'hover:bg-slate-50 text-slate-700'
                    }`}
                  >
                    <td className="py-3.5 px-5 font-sans flex items-center gap-2.5">
                      {isEnsemble ? (
                        <div className="w-2 h-2 rounded-full bg-indigo-500 shrink-0" />
                      ) : (
                        <div className={`w-2 h-2 rounded-full shrink-0 ${isDark ? 'bg-slate-600' : 'bg-slate-300'}`} />
                      )}
                      <span className={isEnsemble ? 'text-indigo-600 dark:text-indigo-300' : ''}>
                        {m.modelName}
                      </span>
                      {isBest && (
                        <span className="px-1.5 py-0.5 text-[10px] rounded bg-amber-500/10 border border-amber-500/30 text-amber-500 font-sans uppercase tracking-wider font-semibold">
                          Best
                        </span>
                      )}
                    </td>
                    <td className="py-3.5 px-5 text-right text-indigo-500 font-semibold text-sm">
                      {m.accuracy.toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-5 text-right">
                      {m.precision.toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-5 text-right">
                      {m.recall.toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-5 text-right text-emerald-500 font-semibold">
                      {m.f1Score.toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-5 text-center font-sans">
                      <span className={`inline-flex items-center text-[11px] ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                        Evaluated ✓
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confusion Matrix Card for Soft Voting Ensemble */}
      {cm && (
        <div
          className={`border rounded-xl p-6 space-y-6 transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <div
            className={`flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b pb-4 ${
              isDark ? 'border-slate-800' : 'border-slate-200'
            }`}
          >
            <div>
              <h2
                className={`text-sm font-semibold tracking-tight ${
                  isDark ? 'text-white' : 'text-slate-900'
                }`}
              >
                Confusion Matrix: Soft Voting Ensemble (Model v{modelVersion})
              </h2>
              <p className={`text-xs mt-0.5 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                Breakdown of {cm.total} test predictions against ground truth labels
              </p>
            </div>
            <div className="text-xs font-mono tabular-nums">
              <span className={isDark ? 'text-slate-400' : 'text-slate-500'}>Accuracy: </span>
              <span className="text-emerald-500 font-semibold">{ensembleMetrics.accuracy}%</span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            {/* 2x2 Matrix Grid */}
            <div className="max-w-md mx-auto w-full">
              <div
                className={`text-center text-xs font-semibold mb-2 ${
                  isDark ? 'text-slate-400' : 'text-slate-500'
                }`}
              >
                Predicted Sentiment
              </div>
              <div className="flex">
                {/* Left vertical label */}
                <div
                  className={`flex items-center justify-center -rotate-90 text-xs font-semibold w-8 shrink-0 ${
                    isDark ? 'text-slate-400' : 'text-slate-500'
                  }`}
                >
                  Actual
                </div>

                <div className="flex-1 space-y-2">
                  {/* Column headers */}
                  <div
                    className={`grid grid-cols-2 gap-2 text-center text-xs font-medium ${
                      isDark ? 'text-slate-400' : 'text-slate-500'
                    }`}
                  >
                    <div>Predicted Positive</div>
                    <div>Predicted Negative</div>
                  </div>

                  {/* Row 1: Actual Positive */}
                  <div className="grid grid-cols-2 gap-2">
                    <div
                      className={`p-4 rounded-lg border text-center ${
                        isDark
                          ? 'bg-emerald-950/40 border-emerald-500/30'
                          : 'bg-emerald-50 border-emerald-200'
                      }`}
                    >
                      <div className="text-[11px] text-emerald-600 dark:text-emerald-300 font-medium">
                        True Positive (TP)
                      </div>
                      <div className="text-2xl font-bold text-emerald-500 font-mono tabular-nums mt-1">
                        {cm.truePositive}
                      </div>
                      <div className={`text-[10px] mt-1 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                        Correctly identified positive
                      </div>
                    </div>

                    <div
                      className={`p-4 rounded-lg border text-center ${
                        isDark
                          ? 'bg-rose-950/20 border-rose-500/20'
                          : 'bg-rose-50 border-rose-200'
                      }`}
                    >
                      <div className="text-[11px] text-rose-600 dark:text-rose-300 font-medium">
                        False Negative (FN)
                      </div>
                      <div className="text-2xl font-bold text-rose-500 font-mono tabular-nums mt-1">
                        {cm.falseNegative}
                      </div>
                      <div className={`text-[10px] mt-1 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                        Positive misclassified as neg
                      </div>
                    </div>
                  </div>

                  {/* Row 2: Actual Negative */}
                  <div className="grid grid-cols-2 gap-2">
                    <div
                      className={`p-4 rounded-lg border text-center ${
                        isDark
                          ? 'bg-rose-950/20 border-rose-500/20'
                          : 'bg-rose-50 border-rose-200'
                      }`}
                    >
                      <div className="text-[11px] text-rose-600 dark:text-rose-300 font-medium">
                        False Positive (FP)
                      </div>
                      <div className="text-2xl font-bold text-rose-500 font-mono tabular-nums mt-1">
                        {cm.falsePositive}
                      </div>
                      <div className={`text-[10px] mt-1 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                        Negative misclassified as pos
                      </div>
                    </div>

                    <div
                      className={`p-4 rounded-lg border text-center ${
                        isDark
                          ? 'bg-emerald-950/40 border-emerald-500/30'
                          : 'bg-emerald-50 border-emerald-200'
                      }`}
                    >
                      <div className="text-[11px] text-emerald-600 dark:text-emerald-300 font-medium">
                        True Negative (TN)
                      </div>
                      <div className="text-2xl font-bold text-emerald-500 font-mono tabular-nums mt-1">
                        {cm.trueNegative}
                      </div>
                      <div className={`text-[10px] mt-1 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                        Correctly identified negative
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Matrix Statistical Breakdown */}
            <div
              className={`space-y-3 p-5 rounded-xl border text-xs ${
                isDark
                  ? 'bg-slate-950/60 border-slate-800'
                  : 'bg-slate-50 border-slate-200'
              }`}
            >
              <div
                className={`text-xs font-semibold mb-2 ${
                  isDark ? 'text-slate-200' : 'text-slate-800'
                }`}
              >
                Performance Summary
              </div>
              
              <div
                className={`flex items-center justify-between py-1.5 border-b ${
                  isDark ? 'border-slate-800' : 'border-slate-200'
                }`}
              >
                <span className={isDark ? 'text-slate-400' : 'text-slate-500'}>
                  Total Test Instances:
                </span>
                <span className="font-mono tabular-nums font-semibold">
                  {cm.total}
                </span>
              </div>

              <div
                className={`flex items-center justify-between py-1.5 border-b ${
                  isDark ? 'border-slate-800' : 'border-slate-200'
                }`}
              >
                <span className={isDark ? 'text-slate-400' : 'text-slate-500'}>
                  Correct Predictions (TP + TN):
                </span>
                <span className="font-mono text-emerald-500 tabular-nums font-semibold">
                  {cm.truePositive + cm.trueNegative} ({(
                    ((cm.truePositive + cm.trueNegative) / cm.total) *
                    100
                  ).toFixed(1)}%)
                </span>
              </div>

              <div
                className={`flex items-center justify-between py-1.5 border-b ${
                  isDark ? 'border-slate-800' : 'border-slate-200'
                }`}
              >
                <span className={isDark ? 'text-slate-400' : 'text-slate-500'}>
                  Classification Errors (FP + FN):
                </span>
                <span className="font-mono text-rose-500 tabular-nums font-semibold">
                  {cm.falsePositive + cm.falseNegative} ({(
                    ((cm.falsePositive + cm.falseNegative) / cm.total) *
                    100
                  ).toFixed(1)}%)
                </span>
              </div>

              <div
                className={`flex items-center justify-between py-1.5 border-b ${
                  isDark ? 'border-slate-800' : 'border-slate-200'
                }`}
              >
                <span className={isDark ? 'text-slate-400' : 'text-slate-500'}>
                  Sensitivity / Recall:
                </span>
                <span className="font-mono text-sky-500 tabular-nums font-semibold">
                  {ensembleMetrics.recall}%
                </span>
              </div>

              <div className="flex items-center justify-between py-1.5">
                <span className={isDark ? 'text-slate-400' : 'text-slate-500'}>
                  Precision:
                </span>
                <span className="font-mono text-indigo-500 tabular-nums font-semibold">
                  {ensembleMetrics.precision}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
