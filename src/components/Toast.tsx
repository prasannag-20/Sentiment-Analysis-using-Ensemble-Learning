import React, { useEffect } from 'react';
import { CheckCircle2, AlertCircle, X } from 'lucide-react';
import { Theme } from '../types';

export interface ToastMessage {
  id: string;
  type: 'success' | 'error';
  title: string;
  message: string;
}

interface ToastProps {
  toast: ToastMessage | null;
  onClose: () => void;
  theme: Theme;
}

export const Toast: React.FC<ToastProps> = ({ toast, onClose, theme }) => {
  useEffect(() => {
    if (!toast) return;

    const timer = setTimeout(() => {
      onClose();
    }, 4500);

    return () => clearTimeout(timer);
  }, [toast, onClose]);

  if (!toast) return null;

  const isDark = theme === 'dark';
  const isSuccess = toast.type === 'success';

  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed top-5 right-5 z-50 max-w-md w-full animate-in fade-in slide-in-from-top-3 duration-300 pointer-events-auto"
    >
      <div
        className={`p-4 rounded-xl border shadow-xl flex items-start gap-3 transition-colors ${
          isSuccess
            ? isDark
              ? 'bg-slate-900/95 border-emerald-500/40 text-slate-100 shadow-emerald-950/20'
              : 'bg-white border-emerald-300 text-slate-900 shadow-emerald-500/10'
            : isDark
            ? 'bg-slate-900/95 border-rose-500/40 text-slate-100 shadow-rose-950/20'
            : 'bg-white border-rose-300 text-slate-900 shadow-rose-500/10'
        }`}
      >
        {/* Status Icon */}
        <div className="shrink-0 mt-0.5">
          {isSuccess ? (
            <div className="w-8 h-8 rounded-lg bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-emerald-500">
              <CheckCircle2 className="w-4 h-4" />
            </div>
          ) : (
            <div className="w-8 h-8 rounded-lg bg-rose-500/15 border border-rose-500/30 flex items-center justify-center text-rose-500">
              <AlertCircle className="w-4 h-4" />
            </div>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0 pr-1">
          <div className="text-xs font-bold leading-tight flex items-center gap-1.5">
            <span className={isSuccess ? 'text-emerald-500' : 'text-rose-500'}>
              {toast.title}
            </span>
          </div>
          <p
            className={`text-xs mt-1 leading-relaxed ${
              isDark ? 'text-slate-300' : 'text-slate-600'
            }`}
          >
            {toast.message}
          </p>
        </div>

        {/* Close Button */}
        <button
          onClick={onClose}
          aria-label="Dismiss notification"
          className={`shrink-0 p-1 rounded-lg transition-colors ${
            isDark
              ? 'text-slate-400 hover:text-white hover:bg-slate-800'
              : 'text-slate-400 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
