import React, { useMemo, useState, useRef } from 'react';
import { 
  BarChart3, 
  ArrowRight,
  Activity,
  Award,
  Eye,
  EyeOff,
  AlertTriangle,
  HelpCircle,
  TrendingUp
} from 'lucide-react';
import { ActiveTab, DatasetStats, EvaluationResults, ModelRocData, RocPoint, Theme } from '../../types';

interface VisualizationSectionProps {
  stats: DatasetStats;
  evaluation: EvaluationResults | null;
  isTrained: boolean;
  modelVersion: number;
  theme: Theme;
  onNavigate: (tab: ActiveTab) => void;
}

export const VisualizationSection: React.FC<VisualizationSectionProps> = ({
  stats,
  evaluation,
  isTrained,
  modelVersion,
  theme,
  onNavigate
}) => {
  const isDark = theme === 'dark';
  const svgRef = useRef<SVGSVGElement | null>(null);

  const [hoveredModelKey, setHoveredModelKey] = useState<string | null>(null);
  const [activeCurves, setActiveCurves] = useState<Record<string, boolean>>({
    ensemble: true,
    logisticRegression: true,
    naiveBayes: true,
    linearSVM: true,
    randomForest: true,
    sgdClassifier: true
  });
  const [hoverPos, setHoverPos] = useState<{ fpr: number; svgX: number; svgY: number } | null>(null);

  const modelMetricsList = useMemo(() => {
    if (!evaluation) return [];
    return [
      { label: 'Logistic Reg.', ...evaluation.models.logisticRegression },
      { label: 'Naive Bayes', ...evaluation.models.naiveBayes },
      { label: 'Linear SVM', ...evaluation.models.linearSVM },
      { label: 'Random Forest', ...evaluation.models.randomForest },
      { label: 'SGD Classifier', ...evaluation.models.sgdClassifier },
      { label: 'Soft Ensemble', ...evaluation.models.ensemble }
    ];
  }, [evaluation]);

  const { posPct, negPct } = useMemo(() => {
    const pos = stats.total > 0 ? (stats.positive / stats.total) * 100 : 50;
    const neg = stats.total > 0 ? (stats.negative / stats.total) * 100 : 50;
    return { posPct: pos, negPct: neg };
  }, [stats.total, stats.positive, stats.negative]);

  if (!isTrained || !evaluation) {
    return (
      <div
        className={`max-w-4xl mx-auto py-12 text-center border rounded-xl p-8 space-y-4 transition-colors ${
          isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
        }`}
      >
        <div className="w-12 h-12 rounded-xl bg-indigo-600/20 text-indigo-500 mx-auto flex items-center justify-center">
          <BarChart3 className="w-6 h-6" />
        </div>
        <h2 className={`text-base font-semibold ${isDark ? 'text-white' : 'text-slate-900'}`}>
          Visualizations Unavailable
        </h2>
        <p className={`text-xs max-w-md mx-auto ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
          Please train the ML models first to generate performance visualizations and confusion matrices.
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

  const cm = evaluation.models.ensemble.confusionMatrix;

  const rocSeriesConfig = useMemo(
    () => [
      { key: 'ensemble' as const, name: 'Soft Voting Ensemble', color: '#6366f1', strokeWidth: 2.8, isEnsemble: true },
      { key: 'logisticRegression' as const, name: 'Logistic Regression', color: '#3b82f6', strokeWidth: 2, isEnsemble: false },
      { key: 'naiveBayes' as const, name: 'Multinomial Naive Bayes', color: '#10b981', strokeWidth: 2, isEnsemble: false },
      { key: 'linearSVM' as const, name: 'Linear SVM', color: '#f59e0b', strokeWidth: 2, isEnsemble: false },
      { key: 'randomForest' as const, name: 'Random Forest', color: '#ec4899', strokeWidth: 2, isEnsemble: false },
      { key: 'sgdClassifier' as const, name: 'SGD Classifier', color: '#06b6d4', strokeWidth: 2, isEnsemble: false }
    ],
    []
  );

  const rocData = evaluation.rocCurve;
  const hasValidRoc = Boolean(rocData && rocData.hasValidData && rocData.models);

  // Find top AUC model
  const topAucModel = useMemo(() => {
    if (!hasValidRoc || !rocData?.models) return null;
    let top: { name: string; auc: number } | null = null;
    for (const conf of rocSeriesConfig) {
      const mData = rocData.models[conf.key];
      if (mData && (top === null || mData.auc > top.auc)) {
        top = { name: mData.modelName, auc: mData.auc };
      }
    }
    return top;
  }, [hasValidRoc, rocData, rocSeriesConfig]);

  const handleSvgMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    const scaleX = 620 / rect.width;
    const scaleY = 440 / rect.height;
    const svgX = clientX * scaleX;
    const svgY = clientY * scaleY;

    if (svgX >= 60 && svgX <= 580 && svgY >= 30 && svgY <= 380) {
      const fpr = Math.max(0, Math.min(1, (svgX - 60) / 520));
      setHoverPos({ fpr, svgX, svgY });
    } else {
      setHoverPos(null);
    }
  };

  const handleSvgMouseLeave = () => {
    setHoverPos(null);
  };

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
              Model Visualizations
            </h1>
            <span className="font-mono text-xs font-bold text-indigo-500 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded">
              Model v{modelVersion}
            </span>
          </div>
          <p className={`text-xs mt-1 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
            Visual comparisons of sentiment distribution, classifier accuracy, evaluation metrics, and ensemble confusion matrix.
          </p>
        </div>
      </div>

      {/* 2x2 Grid of the 4 Required Visualizations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* GRAPH 1: Positive vs Negative Sentiment Distribution */}
        <div
          className={`border rounded-xl p-5 space-y-4 transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <div className="flex items-center justify-between">
            <h2
              className={`text-sm font-semibold tracking-tight ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              1. Sentiment Class Distribution
            </h2>
            <span
              className={`text-xs font-mono tabular-nums ${
                isDark ? 'text-slate-400' : 'text-slate-500'
              }`}
            >
              N = {stats.total.toLocaleString()} reviews
            </span>
          </div>

          <div className="py-4 flex flex-col items-center justify-center">
            {/* Donut Chart SVG */}
            <div className="relative w-44 h-44">
              <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                {/* Background ring */}
                <circle
                  cx="50"
                  cy="50"
                  r="38"
                  fill="none"
                  stroke={isDark ? '#1e293b' : '#e2e8f0'}
                  strokeWidth="12"
                />
                {/* Positive segment */}
                <circle
                  cx="50"
                  cy="50"
                  r="38"
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="12"
                  strokeDasharray={`${(posPct * 2.387).toFixed(1)} 238.7`}
                  strokeDashoffset="0"
                  strokeLinecap="round"
                />
                {/* Negative segment */}
                <circle
                  cx="50"
                  cy="50"
                  r="38"
                  fill="none"
                  stroke="#f43f5e"
                  strokeWidth="12"
                  strokeDasharray={`${(negPct * 2.387).toFixed(1)} 238.7`}
                  strokeDashoffset={`-${(posPct * 2.387).toFixed(1)}`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                <span
                  className={`text-xl font-bold tabular-nums font-mono ${
                    isDark ? 'text-white' : 'text-slate-900'
                  }`}
                >
                  {stats.total.toLocaleString()}
                </span>
                <span className={`text-[10px] font-medium ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                  Total Reviews
                </span>
              </div>
            </div>

            {/* Legend / Bar indicators */}
            <div className="mt-6 w-full grid grid-cols-2 gap-4">
              <div
                className={`p-3 rounded-lg border text-center ${
                  isDark
                    ? 'bg-slate-950/60 border-slate-800'
                    : 'bg-slate-50 border-slate-200'
                }`}
              >
                <div className="flex items-center justify-center gap-1.5 text-xs text-emerald-500 font-medium">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                  <span>Positive</span>
                </div>
                <div
                  className={`mt-1 text-lg font-bold font-mono tabular-nums ${
                    isDark ? 'text-white' : 'text-slate-900'
                  }`}
                >
                  {stats.positive.toLocaleString()}
                </div>
                <div className={`text-[11px] tabular-nums ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                  {posPct.toFixed(1)}% of dataset
                </div>
              </div>

              <div
                className={`p-3 rounded-lg border text-center ${
                  isDark
                    ? 'bg-slate-950/60 border-slate-800'
                    : 'bg-slate-50 border-slate-200'
                }`}
              >
                <div className="flex items-center justify-center gap-1.5 text-xs text-rose-500 font-medium">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                  <span>Negative</span>
                </div>
                <div
                  className={`mt-1 text-lg font-bold font-mono tabular-nums ${
                    isDark ? 'text-white' : 'text-slate-900'
                  }`}
                >
                  {stats.negative.toLocaleString()}
                </div>
                <div className={`text-[11px] tabular-nums ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                  {negPct.toFixed(1)}% of dataset
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* GRAPH 2: Accuracy Comparison of All Models */}
        <div
          className={`border rounded-xl p-5 space-y-4 transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <div className="flex items-center justify-between">
            <h2
              className={`text-sm font-semibold tracking-tight ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              2. Accuracy Comparison Across Models
            </h2>
            <span className={`text-xs font-mono ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Model v{modelVersion} Test Partition
            </span>
          </div>

          <div className="space-y-3 pt-2">
            {modelMetricsList.map((m, idx) => {
              const isEnsemble = m.modelName.includes('Ensemble');
              return (
                <div key={idx} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span
                      className={`font-medium ${
                        isEnsemble
                          ? 'text-indigo-500 font-semibold'
                          : isDark
                          ? 'text-slate-300'
                          : 'text-slate-700'
                      }`}
                    >
                      {m.label} {isEnsemble && '★'}
                    </span>
                    <span
                      className={`font-mono tabular-nums font-semibold ${
                        isDark ? 'text-white' : 'text-slate-900'
                      }`}
                    >
                      {m.accuracy.toFixed(1)}%
                    </span>
                  </div>
                  <div
                    className={`h-3 rounded-full overflow-hidden border ${
                      isDark
                        ? 'bg-slate-950 border-slate-800'
                        : 'bg-slate-100 border-slate-200'
                    }`}
                  >
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        isEnsemble
                          ? 'bg-gradient-to-r from-indigo-500 to-indigo-400'
                          : isDark
                          ? 'bg-slate-600'
                          : 'bg-slate-400'
                      }`}
                      style={{ width: `${Math.max(8, m.accuracy)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* GRAPH 3: Precision vs Recall vs F1-Score Comparison */}
        <div
          className={`border rounded-xl p-5 space-y-4 transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <div className="flex items-center justify-between">
            <h2
              className={`text-sm font-semibold tracking-tight ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              3. Precision vs Recall vs F1-Score
            </h2>
            <div className="flex items-center gap-3 text-[10px]">
              <span className={`flex items-center gap-1 ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>
                <span className="w-2 h-2 rounded bg-indigo-500" /> Precision
              </span>
              <span className={`flex items-center gap-1 ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>
                <span className="w-2 h-2 rounded bg-sky-400" /> Recall
              </span>
              <span className={`flex items-center gap-1 ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>
                <span className="w-2 h-2 rounded bg-emerald-400" /> F1
              </span>
            </div>
          </div>

          {/* Grouped Bar Visualizer */}
          <div className="space-y-3 pt-2">
            {modelMetricsList.map((m, idx) => (
              <div key={idx} className="space-y-1">
                <div
                  className={`flex items-center justify-between text-[11px] ${
                    isDark ? 'text-slate-300' : 'text-slate-700'
                  }`}
                >
                  <span className="font-medium">{m.label}</span>
                  <span
                    className={`font-mono text-[10px] tabular-nums ${
                      isDark ? 'text-slate-400' : 'text-slate-500'
                    }`}
                  >
                    P: {m.precision.toFixed(0)}% · R: {m.recall.toFixed(0)}% · F1: {m.f1Score.toFixed(0)}%
                  </span>
                </div>
                <div
                  className={`grid grid-cols-3 gap-1 h-2 rounded p-0.5 border ${
                    isDark
                      ? 'bg-slate-950/70 border-slate-800'
                      : 'bg-slate-100 border-slate-200'
                  }`}
                >
                  <div className="h-full bg-indigo-500 rounded-sm" style={{ width: `${m.precision}%` }} />
                  <div className="h-full bg-sky-400 rounded-sm" style={{ width: `${m.recall}%` }} />
                  <div className="h-full bg-emerald-400 rounded-sm" style={{ width: `${m.f1Score}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* GRAPH 4: Confusion Matrix Heatmap for Soft Voting Ensemble */}
        <div
          className={`border rounded-xl p-5 space-y-4 transition-colors ${
            isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
          }`}
        >
          <div className="flex items-center justify-between">
            <h2
              className={`text-sm font-semibold tracking-tight ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              4. Ensemble Confusion Matrix Heatmap
            </h2>
            <span className="text-xs text-indigo-500 font-mono font-semibold">
              Acc: {evaluation.models.ensemble.accuracy}%
            </span>
          </div>

          {cm && (
            <div className="pt-2">
              <div className="grid grid-cols-2 gap-3">
                {/* TP */}
                <div
                  className={`p-4 rounded-xl border text-center flex flex-col justify-between ${
                    isDark
                      ? 'bg-emerald-950/30 border-emerald-500/30'
                      : 'bg-emerald-50 border-emerald-200'
                  }`}
                >
                  <span className="text-[11px] text-emerald-600 dark:text-emerald-300 font-medium">
                    True Positive
                  </span>
                  <div className="text-3xl font-bold text-emerald-500 font-mono tabular-nums my-2">
                    {cm.truePositive}
                  </div>
                  <span className={`text-[10px] ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                    {cm.total > 0 ? `${((cm.truePositive / cm.total) * 100).toFixed(1)}% of test samples` : ''}
                  </span>
                </div>

                {/* FN */}
                <div
                  className={`p-4 rounded-xl border text-center flex flex-col justify-between ${
                    isDark
                      ? 'bg-rose-950/20 border-rose-500/20'
                      : 'bg-rose-50 border-rose-200'
                  }`}
                >
                  <span className="text-[11px] text-rose-600 dark:text-rose-300 font-medium">
                    False Negative
                  </span>
                  <div className="text-3xl font-bold text-rose-500 font-mono tabular-nums my-2">
                    {cm.falseNegative}
                  </div>
                  <span className={`text-[10px] ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                    {cm.total > 0 ? `${((cm.falseNegative / cm.total) * 100).toFixed(1)}% of test samples` : ''}
                  </span>
                </div>

                {/* FP */}
                <div
                  className={`p-4 rounded-xl border text-center flex flex-col justify-between ${
                    isDark
                      ? 'bg-rose-950/20 border-rose-500/20'
                      : 'bg-rose-50 border-rose-200'
                  }`}
                >
                  <span className="text-[11px] text-rose-600 dark:text-rose-300 font-medium">
                    False Positive
                  </span>
                  <div className="text-3xl font-bold text-rose-500 font-mono tabular-nums my-2">
                    {cm.falsePositive}
                  </div>
                  <span className={`text-[10px] ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                    {cm.total > 0 ? `${((cm.falsePositive / cm.total) * 100).toFixed(1)}% of test samples` : ''}
                  </span>
                </div>

                {/* TN */}
                <div
                  className={`p-4 rounded-xl border text-center flex flex-col justify-between ${
                    isDark
                      ? 'bg-emerald-950/30 border-emerald-500/30'
                      : 'bg-emerald-50 border-emerald-200'
                  }`}
                >
                  <span className="text-[11px] text-emerald-600 dark:text-emerald-300 font-medium">
                    True Negative
                  </span>
                  <div className="text-3xl font-bold text-emerald-500 font-mono tabular-nums my-2">
                    {cm.trueNegative}
                  </div>
                  <span className={`text-[10px] ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                    {cm.total > 0 ? `${((cm.trueNegative / cm.total) * 100).toFixed(1)}% of test samples` : ''}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* GRAPH 5: ROC Curve (Receiver Operating Characteristic) */}
      <div
        className={`border rounded-xl p-5 md:p-6 space-y-5 transition-colors ${
          isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200 shadow-sm'
        }`}
      >
        {/* Card Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b pb-4 border-inherit">
          <div className="space-y-1">
            <div className="flex items-center gap-2.5">
              <h2
                className={`text-sm md:text-base font-semibold tracking-tight ${
                  isDark ? 'text-white' : 'text-slate-900'
                }`}
              >
                5. ROC Curves (Receiver Operating Characteristic)
              </h2>
              <span className="inline-flex items-center gap-1 font-mono text-[11px] font-bold text-indigo-500 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded">
                <Activity className="w-3 h-3" />
                Held-Out Test Set (N = {evaluation.testSetSize})
              </span>
            </div>
            <p className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              True Positive Rate (TPR) vs. False Positive Rate (FPR) across discrimination thresholds for all 5 base classifiers and the Soft Voting Ensemble.
            </p>
          </div>

          {topAucModel && (
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs shrink-0 ${
                isDark
                  ? 'bg-amber-950/20 border-amber-500/30 text-amber-300'
                  : 'bg-amber-50 border-amber-200 text-amber-800'
              }`}
            >
              <Award className="w-4 h-4 text-amber-500" />
              <div>
                <span className="font-medium">Top AUC: </span>
                <span className="font-bold">{topAucModel.name}</span>{' '}
                <span className="font-mono tabular-nums font-semibold">({topAucModel.auc.toFixed(4)})</span>
              </div>
            </div>
          )}
        </div>

        {!hasValidRoc ? (
          <div className="py-12 px-6 text-center rounded-xl border border-dashed border-amber-500/30 bg-amber-500/5 space-y-3">
            <AlertTriangle className="w-8 h-8 text-amber-500 mx-auto" />
            <h3 className={`text-sm font-semibold ${isDark ? 'text-white' : 'text-slate-900'}`}>
              ROC Curve Data Unavailable
            </h3>
            <p className={`text-xs max-w-md mx-auto ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              {rocData?.errorMessage ||
                'ROC curve generation requires genuine held-out test predictions containing both positive and negative reviews. Please train the models to generate real evaluation curves.'}
            </p>
            <button
              onClick={() => onNavigate('train')}
              className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition-colors cursor-pointer"
            >
              <span>Go to Train Model</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column: Responsive SVG Chart */}
            <div className="lg:col-span-8 flex flex-col items-center">
              <div className="w-full relative">
                <svg
                  ref={svgRef}
                  viewBox="0 0 620 440"
                  className="w-full h-auto select-none"
                  onMouseMove={handleSvgMouseMove}
                  onMouseLeave={handleSvgMouseLeave}
                >
                  {/* Chart Background */}
                  <rect
                    x="60"
                    y="30"
                    width="520"
                    height="350"
                    fill={isDark ? '#0b0f19' : '#f8fafc'}
                    rx="6"
                    stroke={isDark ? '#1e293b' : '#e2e8f0'}
                    strokeWidth="1"
                  />

                  {/* Grid Lines & Ticks */}
                  {[0.0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tick) => {
                    const x = 60 + tick * 520;
                    const y = 30 + (1 - tick) * 350;
                    return (
                      <g key={tick}>
                        {/* Vertical Grid Line */}
                        <line
                          x1={x}
                          y1="30"
                          x2={x}
                          y2="380"
                          stroke={isDark ? '#1e293b' : '#e2e8f0'}
                          strokeWidth="1"
                          strokeDasharray={tick > 0 && tick < 1 ? '2 2' : undefined}
                        />
                        {/* Horizontal Grid Line */}
                        <line
                          x1="60"
                          y1={y}
                          x2="580"
                          y2={y}
                          stroke={isDark ? '#1e293b' : '#e2e8f0'}
                          strokeWidth="1"
                          strokeDasharray={tick > 0 && tick < 1 ? '2 2' : undefined}
                        />
                        {/* X-axis Tick Label */}
                        <text
                          x={x}
                          y="400"
                          textAnchor="middle"
                          fontSize="10"
                          fontFamily="monospace"
                          fill={isDark ? '#94a3b8' : '#64748b'}
                        >
                          {tick.toFixed(1)}
                        </text>
                        {/* Y-axis Tick Label */}
                        <text
                          x="50"
                          y={y + 3.5}
                          textAnchor="end"
                          fontSize="10"
                          fontFamily="monospace"
                          fill={isDark ? '#94a3b8' : '#64748b'}
                        >
                          {tick.toFixed(1)}
                        </text>
                      </g>
                    );
                  })}

                  {/* Diagonal Reference Line for Random Classifier (FPR = TPR, AUC = 0.50) */}
                  <line
                    x1="60"
                    y1="380"
                    x2="580"
                    y2="30"
                    stroke={isDark ? '#475569' : '#94a3b8'}
                    strokeWidth="1.5"
                    strokeDasharray="4 4"
                  />

                  {/* ROC Curves for Each Classifier */}
                  {rocSeriesConfig.map((conf) => {
                    const mData = rocData?.models?.[conf.key as keyof typeof rocData.models];
                    if (!mData || !activeCurves[conf.key]) return null;

                    const isHovered = hoveredModelKey === conf.key;
                    const opacity = hoveredModelKey ? (isHovered ? 1 : 0.22) : 0.95;
                    const strokeW = isHovered ? conf.strokeWidth + 1.2 : conf.strokeWidth;

                    const pathStr = mData.points
                      .map(
                        (p, i) =>
                          `${i === 0 ? 'M' : 'L'} ${(60 + p.fpr * 520).toFixed(1)} ${(
                            30 +
                            (1 - p.tpr) * 350
                          ).toFixed(1)}`
                      )
                      .join(' ');

                    return (
                      <g key={conf.key} className="transition-opacity duration-150">
                        <path
                          d={pathStr}
                          fill="none"
                          stroke={conf.color}
                          strokeWidth={strokeW}
                          strokeOpacity={opacity}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        {/* Small vertex circles for clarity */}
                        {mData.points.map((pt, pIdx) => (
                          <circle
                            key={pIdx}
                            cx={60 + pt.fpr * 520}
                            cy={30 + (1 - pt.tpr) * 350}
                            r={isHovered ? 3.5 : conf.isEnsemble ? 2.5 : 2}
                            fill={conf.color}
                            opacity={opacity}
                          />
                        ))}
                      </g>
                    );
                  })}

                  {/* Crosshair when hovering over the plot */}
                  {hoverPos && (
                    <g>
                      {/* Vertical line at hover FPR */}
                      <line
                        x1={hoverPos.svgX}
                        y1="30"
                        x2={hoverPos.svgX}
                        y2="380"
                        stroke={isDark ? '#64748b' : '#94a3b8'}
                        strokeWidth="1"
                        strokeDasharray="3 3"
                      />

                      {/* Highlighted circles on each visible curve at the hover FPR */}
                      {rocSeriesConfig.map((conf) => {
                        const mData = rocData?.models?.[conf.key as keyof typeof rocData.models];
                        if (!mData || !activeCurves[conf.key]) return null;

                        // Find closest point
                        let closest = mData.points[0];
                        let minDiff = Math.abs(mData.points[0].fpr - hoverPos.fpr);
                        for (let i = 1; i < mData.points.length; i++) {
                          const diff = Math.abs(mData.points[i].fpr - hoverPos.fpr);
                          if (diff < minDiff) {
                            minDiff = diff;
                            closest = mData.points[i];
                          }
                        }

                        const cx = 60 + closest.fpr * 520;
                        const cy = 30 + (1 - closest.tpr) * 350;

                        return (
                          <circle
                            key={`hover-${conf.key}`}
                            cx={cx}
                            cy={cy}
                            r="5"
                            fill={conf.color}
                            stroke={isDark ? '#0f172a' : '#ffffff'}
                            strokeWidth="2"
                          />
                        );
                      })}
                    </g>
                  )}

                  {/* Axis Title Labels */}
                  <text
                    x="320"
                    y="428"
                    textAnchor="middle"
                    fontSize="11"
                    fontWeight="600"
                    fill={isDark ? '#cbd5e1' : '#334155'}
                  >
                    False Positive Rate (FPR = 1 - Specificity)
                  </text>

                  <text
                    x="-205"
                    y="22"
                    textAnchor="middle"
                    fontSize="11"
                    fontWeight="600"
                    transform="rotate(-90)"
                    fill={isDark ? '#cbd5e1' : '#334155'}
                  >
                    True Positive Rate (TPR = Sensitivity / Recall)
                  </text>
                </svg>

                {/* Floating Inspection Tooltip when Hovering */}
                {hoverPos && (
                  <div
                    className={`mt-2 p-2.5 rounded-lg border text-xs shadow-md flex flex-wrap items-center justify-between gap-3 ${
                      isDark ? 'bg-slate-950/90 border-slate-700 text-white' : 'bg-white border-slate-200 text-slate-800'
                    }`}
                  >
                    <div className="font-mono text-[11px] font-semibold text-indigo-400">
                      FPR: {(hoverPos.fpr * 100).toFixed(1)}% ({hoverPos.fpr.toFixed(3)})
                    </div>
                    <div className="flex flex-wrap items-center gap-3 text-[11px]">
                      {rocSeriesConfig.map((conf) => {
                        const mData = rocData?.models?.[conf.key as keyof typeof rocData.models];
                        if (!mData || !activeCurves[conf.key]) return null;
                        let closest = mData.points[0];
                        let minDiff = Math.abs(mData.points[0].fpr - hoverPos.fpr);
                        for (let i = 1; i < mData.points.length; i++) {
                          const diff = Math.abs(mData.points[i].fpr - hoverPos.fpr);
                          if (diff < minDiff) {
                            minDiff = diff;
                            closest = mData.points[i];
                          }
                        }
                        return (
                          <span key={conf.key} className="flex items-center gap-1 font-mono">
                            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: conf.color }} />
                            <span className="font-medium">{conf.name.replace('Multinomial ', '').replace('Classifier', '').replace('Regression', 'Reg.')}:</span>
                            <span className="font-bold">TPR {(closest.tpr * 100).toFixed(0)}%</span>
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: Interactive Legend & AUC Table */}
            <div className="lg:col-span-4 space-y-3">
              <div className="flex items-center justify-between pb-1 border-b border-inherit">
                <span className={`text-xs font-semibold ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                  Model Curves &amp; AUC
                </span>
                <div className="flex items-center gap-2 text-[10px]">
                  <button
                    onClick={() => {
                      const allOn: Record<string, boolean> = {};
                      rocSeriesConfig.forEach((c) => (allOn[c.key] = true));
                      setActiveCurves(allOn);
                    }}
                    className={`hover:underline cursor-pointer ${
                      isDark ? 'text-indigo-400' : 'text-indigo-600'
                    }`}
                  >
                    Show all
                  </button>
                  <span className={isDark ? 'text-slate-600' : 'text-slate-300'}>|</span>
                  <button
                    onClick={() => {
                      const allOff: Record<string, boolean> = {};
                      rocSeriesConfig.forEach((c) => (allOff[c.key] = false));
                      allOff.ensemble = true;
                      setActiveCurves(allOff);
                    }}
                    className={`hover:underline cursor-pointer ${
                      isDark ? 'text-indigo-400' : 'text-indigo-600'
                    }`}
                  >
                    Ensemble only
                  </button>
                </div>
              </div>

              {/* Model List Cards */}
              <div className="space-y-2">
                {rocSeriesConfig.map((conf) => {
                  const mData = rocData?.models?.[conf.key as keyof typeof rocData.models];
                  const isVisible = activeCurves[conf.key] ?? true;
                  const isHovered = hoveredModelKey === conf.key;
                  const isTop = topAucModel?.name === mData?.modelName;

                  return (
                    <div
                      key={conf.key}
                      onMouseEnter={() => setHoveredModelKey(conf.key)}
                      onMouseLeave={() => setHoveredModelKey(null)}
                      className={`p-2.5 rounded-lg border text-xs transition-all ${
                        isHovered
                          ? isDark
                            ? 'bg-slate-800/80 border-slate-600 shadow-sm'
                            : 'bg-slate-50 border-slate-300 shadow-sm'
                          : isDark
                          ? 'bg-slate-950/60 border-slate-800'
                          : 'bg-white border-slate-200'
                      } ${!isVisible ? 'opacity-40' : 'opacity-100'}`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span
                            className="w-3 h-3 rounded-full shrink-0 shadow-sm"
                            style={{ backgroundColor: conf.color }}
                          />
                          <div>
                            <div className="flex items-center gap-1.5 font-medium">
                              <span className={isDark ? 'text-white' : 'text-slate-900'}>
                                {conf.name}
                              </span>
                              {conf.isEnsemble && (
                                <span className="px-1 py-0.2 rounded text-[9px] font-bold bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                                  Ensemble
                                </span>
                              )}
                              {isTop && (
                                <span className="px-1 py-0.2 rounded text-[9px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30">
                                  Top AUC
                                </span>
                              )}
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          <span className="font-mono tabular-nums font-bold text-xs text-indigo-500 dark:text-indigo-400">
                            AUC {mData ? mData.auc.toFixed(4) : 'N/A'}
                          </span>
                          <button
                            onClick={() =>
                              setActiveCurves((prev) => ({
                                ...prev,
                                [conf.key]: !prev[conf.key]
                              }))
                            }
                            className="p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
                            title={isVisible ? 'Hide curve' : 'Show curve'}
                          >
                            {isVisible ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                          </button>
                        </div>
                      </div>

                      {/* Mini AUC Progress Bar */}
                      {mData && (
                        <div className="mt-2 flex items-center gap-2">
                          <div
                            className={`flex-1 h-1.5 rounded-full overflow-hidden ${
                              isDark ? 'bg-slate-800' : 'bg-slate-100'
                            }`}
                          >
                            <div
                              className="h-full rounded-full transition-all duration-500"
                              style={{
                                width: `${Math.max(5, mData.auc * 100)}%`,
                                backgroundColor: conf.color
                              }}
                            />
                          </div>
                          <span className={`text-[10px] font-mono tabular-nums ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                            {(mData.auc * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                    </div>
                  );
                })}

                {/* Random Classifier Reference Indicator */}
                <div
                  className={`p-2.5 rounded-lg border text-xs flex items-center justify-between ${
                    isDark ? 'bg-slate-950/40 border-slate-800/80 text-slate-400' : 'bg-slate-50 border-slate-200 text-slate-500'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className="w-4 border-t-2 border-dashed border-slate-400 shrink-0" />
                    <span className="font-medium text-[11px]">Random Classifier (Diagonal)</span>
                  </div>
                  <span className="font-mono text-[11px] font-medium">AUC 0.5000</span>
                </div>
              </div>

              {/* Explanatory Note */}
              <div
                className={`p-3 rounded-lg border text-[11px] space-y-1.5 ${
                  isDark ? 'bg-slate-950/40 border-slate-800 text-slate-400' : 'bg-slate-50 border-slate-200 text-slate-600'
                }`}
              >
                <div className="flex items-center gap-1.5 font-semibold text-slate-300 dark:text-slate-200">
                  <TrendingUp className="w-3.5 h-3.5 text-indigo-400" />
                  <span>ROC &amp; AUC Interpretation</span>
                </div>
                <p>
                  <strong>AUC &gt; 0.90</strong> indicates outstanding classification discrimination. Curves hugging the upper-left corner demonstrate high sensitivity (TPR) while sustaining low false alarm rates (FPR).
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
