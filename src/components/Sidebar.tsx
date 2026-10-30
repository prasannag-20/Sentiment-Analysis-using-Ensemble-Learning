import React from 'react';
import { 
  LayoutDashboard, 
  Bot, 
  FolderGit2, 
  BrainCircuit, 
  BarChart2, 
  LineChart,
  Layers
} from 'lucide-react';
import { ActiveTab, Theme } from '../types';

interface SidebarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  datasetName: string;
  isTrained: boolean;
  modelVersion: number;
  datasetsCount: number;
  theme: Theme;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  datasetName,
  isTrained,
  modelVersion,
  datasetsCount,
  theme
}) => {
  const isDark = theme === 'dark';

  const navItems: { id: ActiveTab; label: string; emoji: string; icon: React.ElementType }[] = [
    { id: 'dashboard', label: 'Dashboard', emoji: '🏠', icon: LayoutDashboard },
    { id: 'chatbot', label: 'Sentiment Chatbot', emoji: '🤖', icon: Bot },
    { id: 'dataset', label: 'Dataset', emoji: '📂', icon: FolderGit2 },
    { id: 'train', label: 'Train Model', emoji: '🧠', icon: BrainCircuit },
    { id: 'evaluation', label: 'Evaluation', emoji: '📊', icon: BarChart2 },
    { id: 'visualization', label: 'Visualization', emoji: '📈', icon: LineChart }
  ];

  return (
    <aside
      className={`w-64 shrink-0 flex flex-col justify-between min-h-screen transition-colors duration-200 border-r ${
        isDark
          ? 'bg-slate-900 border-slate-800 text-slate-200'
          : 'bg-white border-slate-200 text-slate-800'
      }`}
    >
      <div className="p-4">
        {/* Brand Header */}
        <div
          className={`flex items-center gap-3 px-2 py-3 mb-6 border-b ${
            isDark ? 'border-slate-800' : 'border-slate-200'
          }`}
        >
          <div className="w-9 h-9 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-500 shrink-0">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h1
              className={`text-sm font-bold tracking-tight leading-tight ${
                isDark ? 'text-white' : 'text-slate-900'
              }`}
            >
              Sentiment Ensemble
            </h1>
            <p className={`text-[11px] mt-0.5 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Continuous ML Learning
            </p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="space-y-1">
          {navItems.map(item => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-xs font-medium transition-colors text-left cursor-pointer ${
                  isActive
                    ? 'bg-indigo-600 text-white font-semibold shadow-sm'
                    : isDark
                    ? 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/70'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <span className="text-sm shrink-0">{item.emoji}</span>
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Dataset & Model Version Footer */}
      <div
        className={`p-4 border-t text-xs space-y-2 transition-colors ${
          isDark
            ? 'border-slate-800 bg-slate-950/40 text-slate-300'
            : 'border-slate-200 bg-slate-50 text-slate-700'
        }`}
      >
        <div className="flex items-center justify-between text-[11px]">
          <span className={isDark ? 'text-slate-400' : 'text-slate-500'}>Model Engine</span>
          <span className="font-mono font-semibold text-indigo-500 bg-indigo-500/10 px-1.5 py-0.5 rounded text-[10px]">
            Model v{modelVersion}
          </span>
        </div>

        <div className="text-[11px] truncate font-mono" title={datasetName}>
          <span className={isDark ? 'text-slate-400' : 'text-slate-500'}>Dataset: </span>
          <span className={isDark ? 'text-slate-200' : 'text-slate-900'}>{datasetName}</span>
        </div>

        <div className="flex items-center justify-between pt-1 text-[11px]">
          <span className={isDark ? 'text-slate-400' : 'text-slate-500'}>
            Accumulated: {datasetsCount} {datasetsCount === 1 ? 'batch' : 'batches'}
          </span>
          <div className="flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                isTrained ? 'bg-emerald-500 ring-2 ring-emerald-500/20' : 'bg-amber-500 ring-2 ring-amber-500/20'
              }`}
            />
            <span className={isDark ? 'text-slate-300' : 'text-slate-700'}>
              {isTrained ? 'Trained' : 'Needs Train'}
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
};
