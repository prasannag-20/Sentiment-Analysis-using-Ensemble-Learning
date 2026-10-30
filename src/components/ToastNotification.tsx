import React, { useEffect } from 'react';
import { CheckCircle2, AlertCircle, X } from 'lucide-react';
import { Theme } from '../types';

export interface ToastData {
  id: number;
  type: 'success' | 'error';
  title: string;
  message: string;
}

interface ToastNotificationProps {
  toast: ToastData | null;
  onClose: () => void;
  theme: Theme;
}

export const ToastNotification: React.FC<ToastNotificationProps> = ({
  toast,
  onClose,
  theme
}) => {
  useEffect(() => {
    if (!toast) return;

    // Automatically dismiss toast after 4.5 seconds
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
      role="alert"
      aria-live="assertive"
      className="fixed bottom-6 right-6 z-50 max-w-md w-full sm:w-auto animate-in fade-in slide-in-from-bottom-5 duration-300"
    >
      <div
        className={`flex items-start gap-3.5 p-4 rounded-xl border shadow-xl transition-all ${
          isSuccess
            ? isDark
              ? 'bg-slate-900/95 border-emerald-500/40 text-slate-100 shadow-emerald-950/30'
              : 'bg-white border-emerald-500/30 text-slate-900 shadow-emerald-500/10'
            : isDark
            ? 'bg-slate-900/95 border-rose-500/40 text-slate-100 shadow-rose-950/30'
            : 'bg-white border-rose-500/30 text-slate-900 shadow-rose-500/10'
        }`}
      >
        <div
          className={`p-2 rounded-lg shrink-0 mt-0.5 ${
            isSuccess
              ? 'bg-emerald-500/15 text-emerald-500'
              : 'bg-rose-500/15 text-rose-500'
          }`}
        >
          {isSuccess ? (
            <CheckCircle2 className="w-5 h-5" />
          ) : (
            <AlertCircle className="w-5 h-5" />
          )}
        </div>

        <div className="flex-1 pr-2 min-w-0">
          <h4
            className={`text-sm font-semibold tracking-tight ${
              isSuccess
                ? isDark
                  ? 'text-emerald-400'
                  : 'text-emerald-700'
                : isDark
                ? 'text-rose-400'
                : 'text-rose-700'
            }`}
          >
            {toast.title}
          </h4>
          <p
            className={`text-xs mt-1 leading-relaxed ${
              isDark ? 'text-slate-300' : 'text-slate-600'
            }`}
          >
            {toast.message}
          </p>
        </div>

        <button
          onClick={onClose}
          aria-label="Dismiss notification"
          className={`shrink-0 p-1.5 rounded-lg transition-colors cursor-pointer ${
            isDark
              ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              : 'text-slate-400 hover:text-slate-600 hover:bg-slate-100'
          }`}
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
