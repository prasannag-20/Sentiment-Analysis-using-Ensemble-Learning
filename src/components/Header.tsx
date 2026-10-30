import React from 'react';
import { ActiveTab, Theme } from '../types';
import { Bot, Cpu, Sun, Moon } from 'lucide-react';

interface HeaderProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  datasetName: string;
  isTrained: boolean;
  modelVersion: number;
  theme: Theme;
  onToggleTheme: () => void;
}

const tabTitles: Record<ActiveTab, { title: string; subtitle: string }> = {
  dashboard: { title: 'Dashboard', subtitle: 'Overview & Performance Summary' },
  chatbot: { title: 'Sentiment Chatbot', subtitle: 'Interactive Model Inference' },
  dataset: { title: 'Dataset', subtitle: 'Accumulated Benchmark & Batch Processing' },
  train: { title: 'Train Model', subtitle: 'Cumulative Pipeline & Ensemble Training' },
  evaluation: { title: 'Evaluation', subtitle: 'Metrics & Confusion Matrix' },
  visualization: { title: 'Visualization', subtitle: 'Performance & Distribution Charts' }
};

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  datasetName,
  isTrained,
  modelVersion,
  theme,
  onToggleTheme
}) => {
  const current = tabTitles[activeTab];
  const isDark = theme === 'dark';

  return (
    <header
      className={`h-16 px-6 border-b flex items-center justify-between sticky top-0 z-30 transition-colors duration-200 ${
        isDark
          ? 'bg-slate-900/90 backdrop-blur border-slate-800 text-slate-100'
          : 'bg-white/90 backdrop-blur border-slate-200 text-slate-900'
      }`}
    >
      {/* Zone 1: Contextual Breadcrumb & Title */}
      <div className="flex items-center gap-3">
        <span className={`text-xs font-medium ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
          Sentiment Pipeline
        </span>
        <span className={isDark ? 'text-slate-600' : 'text-slate-300'}>/</span>
        <span
          className={`text-sm font-semibold tracking-tight ${
            isDark ? 'text-white' : 'text-slate-900'
          }`}
        >
          {current.title}
        </span>
        <span className="ml-2 font-mono text-[10px] font-semibold text-indigo-500 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded">
          Model v{modelVersion}
        </span>
      </div>

      {/* Zone 2: Shortcuts, Train Button & Theme Toggle */}
      <div className="flex items-center gap-3">
        <span
          className={`hidden lg:inline-flex items-center gap-2 text-xs ${
            isDark ? 'text-slate-400' : 'text-slate-500'
          }`}
        >
          <span>Dataset:</span>
          <span
            className={`font-mono text-[11px] truncate max-w-[180px] ${
              isDark ? 'text-slate-200' : 'text-slate-700'
            }`}
            title={datasetName}
          >
            {datasetName}
          </span>
        </span>

        {activeTab !== 'train' && (
          <button
            onClick={() => setActiveTab('train')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors whitespace-nowrap cursor-pointer ${
              isDark
                ? 'bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700'
                : 'bg-slate-100 hover:bg-slate-200 text-slate-800 border-slate-300'
            }`}
          >
            <Cpu className="w-3.5 h-3.5 text-indigo-500" />
            <span>Train Engine</span>
          </button>
        )}

        {activeTab !== 'chatbot' && (
          <button
            onClick={() => setActiveTab('chatbot')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition-colors whitespace-nowrap shadow-sm cursor-pointer"
          >
            <Bot className="w-3.5 h-3.5" />
            <span>Open Chatbot</span>
          </button>
        )}

        {/* ☀ Light / 🌙 Dark Theme Toggle */}
        <button
          onClick={onToggleTheme}
          aria-label="Toggle theme"
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all duration-200 cursor-pointer ${
            isDark
              ? 'bg-slate-800 hover:bg-slate-700 text-amber-300 border-slate-700 shadow-sm'
              : 'bg-slate-100 hover:bg-slate-200 text-indigo-700 border-slate-300 shadow-sm'
          }`}
          title={`Switch to ${isDark ? 'Light' : 'Dark'} Theme`}
        >
          {isDark ? (
            <>
              <Sun className="w-3.5 h-3.5 text-amber-400" />
              <span>☀ Light</span>
            </>
          ) : (
            <>
              <Moon className="w-3.5 h-3.5 text-indigo-600" />
              <span>🌙 Dark</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
};
