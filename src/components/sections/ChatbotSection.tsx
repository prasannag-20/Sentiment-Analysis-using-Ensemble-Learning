import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Bot, 
  User, 
  CheckCircle2, 
  XCircle, 
  ChevronDown, 
  ChevronUp, 
  RotateCcw,
  Sparkles,
  Loader2
} from 'lucide-react';
import { ChatMessage, PredictionResult, Theme } from '../../types';
import { sentimentEngine } from '../../ml/engine';

interface ChatbotSectionProps {
  isTrained: boolean;
  modelVersion: number;
  theme: Theme;
}

const PRESET_EXAMPLES = [
  "The movie was amazing and I really enjoyed it.",
  "An absolute waste of time with dreadful acting and predictable plot.",
  "The cinematography was breathtaking, although the pacing dragged slightly in the middle.",
  "One of the worst films I have ever seen. Completely boring.",
  "Brilliant performance by the lead cast and exceptional direction throughout."
];

export const ChatbotSection: React.FC<ChatbotSectionProps> = ({
  isTrained,
  modelVersion,
  theme
}) => {
  const isDark = theme === 'dark';

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'bot',
      text: 'Welcome to the Sentiment Inference Chatbot. Type or paste any movie review below to evaluate sentiment across all 5 ML classifiers and the Soft Voting Ensemble.',
      timestamp: 'Ready',
      modelVersion
    }
  ]);

  const [inputReview, setInputReview] = useState('');
  const [isPredictionLoading, setIsPredictionLoading] = useState(false);
  const [expandedDetails, setExpandedDetails] = useState<Record<string, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isPredictionLoading]);

  const handleSend = async (textToSend?: string) => {
    const text = (textToSend || inputReview).trim();
    if (!text || isPredictionLoading) return;

    if (!isTrained) {
      const userMsg: ChatMessage = {
        id: `usr-${Date.now()}`,
        sender: 'user',
        text,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      const botErr: ChatMessage = {
        id: `bot-${Date.now() + 1}`,
        sender: 'bot',
        text: 'The machine learning models are not yet trained. Please go to the Train Model section and click "Train / Update Models" first.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        modelVersion
      };
      setMessages(prev => [...prev, userMsg, botErr]);
      setInputReview('');
      return;
    }

    const userMsg: ChatMessage = {
      id: `usr-${Date.now()}`,
      sender: 'user',
      text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMsg]);
    setInputReview('');
    setIsPredictionLoading(true);

    try {
      // Yield to let React render the user message and show prediction loading state
      await new Promise(r => setTimeout(r, 20));

      // Run prediction using the genuine trained ML pipeline engine (always uses latest model)
      const prediction: PredictionResult = sentimentEngine.predict(text);

      const botMsg: ChatMessage = {
        id: `bot-${Date.now() + 1}`,
        sender: 'bot',
        text: `Sentiment: ${prediction.ensemble.sentiment === 'positive' ? 'Positive' : 'Negative'} (Confidence: ${prediction.ensemble.confidence}%)`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        prediction,
        modelVersion
      };

      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      console.error('Prediction failed:', err);
      const botErrMsg: ChatMessage = {
        id: `bot-${Date.now() + 1}`,
        sender: 'bot',
        text: 'Failed to predict sentiment for this review. Please check the model and try again.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        modelVersion
      };
      setMessages(prev => [...prev, botErrMsg]);
    } finally {
      setIsPredictionLoading(false);
    }
  };

  const toggleDetails = (msgId: string) => {
    setExpandedDetails(prev => ({
      ...prev,
      [msgId]: !prev[msgId]
    }));
  };

  const handleClearChat = () => {
    setMessages([
      {
        id: 'welcome',
        sender: 'bot',
        text: 'Chat history cleared. Enter a movie review to predict sentiment using the trained ensemble system.',
        timestamp: 'Ready',
        modelVersion
      }
    ]);
  };

  return (
    <div className="max-w-4xl mx-auto flex flex-col h-[calc(100vh-8.5rem)]">
      {/* Chat Header */}
      <div
        className={`border rounded-t-xl px-5 py-3.5 flex items-center justify-between shrink-0 transition-colors ${
          isDark
            ? 'bg-slate-900 border-slate-800 text-white'
            : 'bg-white border-slate-200 text-slate-900 shadow-sm'
        }`}
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-500">
            <Bot className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold tracking-tight">
                Sentiment Analysis Chatbot
              </h2>
              {/* Requested Model Version Status */}
              <span className="font-mono text-[10px] font-semibold text-indigo-500 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded">
                Model v{modelVersion}
              </span>
            </div>
            <p className={`text-[11px] mt-0.5 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
              Using latest trained model · 5 Classifiers & Soft Voting Ensemble
            </p>
          </div>
        </div>

        <button
          onClick={handleClearChat}
          className={`text-xs p-1.5 rounded transition-colors ${
            isDark
              ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              : 'text-slate-500 hover:text-slate-800 hover:bg-slate-100'
          }`}
          title="Clear Conversation"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* Messages Scroll Area */}
      <div
        className={`flex-1 border-x p-4 overflow-y-auto space-y-4 transition-colors ${
          isDark
            ? 'bg-slate-950/70 border-slate-800'
            : 'bg-slate-50/80 border-slate-200'
        }`}
      >
        {messages.map(msg => {
          const isUser = msg.sender === 'user';
          const isExpanded = expandedDetails[msg.id] ?? true;

          return (
            <div
              key={msg.id}
              className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
            >
              {!isUser && (
                <div className="w-7 h-7 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-500 shrink-0 mt-0.5">
                  <Bot className="w-3.5 h-3.5" />
                </div>
              )}

              <div
                className={`max-w-[85%] rounded-xl p-4 text-xs transition-colors ${
                  isUser
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : isDark
                    ? 'bg-slate-900 border border-slate-800 text-slate-200'
                    : 'bg-white border border-slate-200 text-slate-800 shadow-sm'
                }`}
              >
                {/* Header/Timestamp */}
                <div className="flex items-center justify-between gap-4 mb-1 text-[11px] opacity-75">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">{isUser ? 'You' : 'Sentiment ML Engine'}</span>
                    {!isUser && msg.modelVersion && (
                      <span className="font-mono text-[10px] text-indigo-400">
                        (Model v{msg.modelVersion})
                      </span>
                    )}
                  </div>
                  <span className="tabular-nums">{msg.timestamp}</span>
                </div>

                {/* Message Text */}
                <div className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</div>

                {/* Prediction Breakdown if present */}
                {msg.prediction && (
                  <div
                    className={`mt-3 pt-3 border-t space-y-3 ${
                      isDark ? 'border-slate-800' : 'border-slate-100'
                    }`}
                  >
                    {/* Primary Ensemble Result Card */}
                    <div
                      className={`flex items-center justify-between p-2.5 rounded-lg border ${
                        isDark
                          ? 'bg-slate-950/80 border-slate-800'
                          : 'bg-slate-50 border-slate-200'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        {msg.prediction.ensemble.sentiment === 'positive' ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                        ) : (
                          <XCircle className="w-4 h-4 text-rose-500" />
                        )}
                        <span className={`font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>
                          Soft Voting Ensemble
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span
                          className={`font-semibold capitalize ${
                            msg.prediction.ensemble.sentiment === 'positive'
                              ? 'text-emerald-500'
                              : 'text-rose-500'
                          }`}
                        >
                          {msg.prediction.ensemble.sentiment}
                        </span>
                        <span className={`font-mono tabular-nums ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                          {msg.prediction.ensemble.confidence}% confidence
                        </span>
                      </div>
                    </div>

                    {/* Toggle Breakdown */}
                    <div className="flex items-center justify-between">
                      <span className={`text-[11px] font-medium ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                        Individual Classifier Predictions:
                      </span>
                      <button
                        onClick={() => toggleDetails(msg.id)}
                        className="text-[11px] text-indigo-500 hover:text-indigo-600 flex items-center gap-0.5 font-medium"
                      >
                        <span>{isExpanded ? 'Hide' : 'Show'} Details</span>
                        {isExpanded ? (
                          <ChevronUp className="w-3 h-3" />
                        ) : (
                          <ChevronDown className="w-3 h-3" />
                        )}
                      </button>
                    </div>

                    {/* Classifier List */}
                    {isExpanded && (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 font-mono text-[11px]">
                        {/* Logistic Regression */}
                        <div
                          className={`p-2 rounded border flex items-center justify-between ${
                            isDark
                              ? 'bg-slate-950/50 border-slate-800'
                              : 'bg-slate-50/70 border-slate-200'
                          }`}
                        >
                          <span className={`font-sans ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                            Logistic Regression
                          </span>
                          <span
                            className={
                              msg.prediction.logisticRegression.sentiment === 'positive'
                                ? 'text-emerald-500 font-semibold'
                                : 'text-rose-500 font-semibold'
                            }
                          >
                            {msg.prediction.logisticRegression.sentiment} ({msg.prediction.logisticRegression.confidence}%)
                          </span>
                        </div>

                        {/* Multinomial Naive Bayes */}
                        <div
                          className={`p-2 rounded border flex items-center justify-between ${
                            isDark
                              ? 'bg-slate-950/50 border-slate-800'
                              : 'bg-slate-50/70 border-slate-200'
                          }`}
                        >
                          <span className={`font-sans ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                            Naive Bayes
                          </span>
                          <span
                            className={
                              msg.prediction.naiveBayes.sentiment === 'positive'
                                ? 'text-emerald-500 font-semibold'
                                : 'text-rose-500 font-semibold'
                            }
                          >
                            {msg.prediction.naiveBayes.sentiment} ({msg.prediction.naiveBayes.confidence}%)
                          </span>
                        </div>

                        {/* Linear SVM */}
                        <div
                          className={`p-2 rounded border flex items-center justify-between ${
                            isDark
                              ? 'bg-slate-950/50 border-slate-800'
                              : 'bg-slate-50/70 border-slate-200'
                          }`}
                        >
                          <span className={`font-sans ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                            Linear SVM
                          </span>
                          <span
                            className={
                              msg.prediction.linearSVM.sentiment === 'positive'
                                ? 'text-emerald-500 font-semibold'
                                : 'text-rose-500 font-semibold'
                            }
                          >
                            {msg.prediction.linearSVM.sentiment} ({msg.prediction.linearSVM.confidence}%)
                          </span>
                        </div>

                        {/* Random Forest */}
                        <div
                          className={`p-2 rounded border flex items-center justify-between ${
                            isDark
                              ? 'bg-slate-950/50 border-slate-800'
                              : 'bg-slate-50/70 border-slate-200'
                          }`}
                        >
                          <span className={`font-sans ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                            Random Forest
                          </span>
                          <span
                            className={
                              msg.prediction.randomForest.sentiment === 'positive'
                                ? 'text-emerald-500 font-semibold'
                                : 'text-rose-500 font-semibold'
                            }
                          >
                            {msg.prediction.randomForest.sentiment} ({msg.prediction.randomForest.confidence}%)
                          </span>
                        </div>

                        {/* SGD Classifier */}
                        <div
                          className={`p-2 rounded border flex items-center justify-between sm:col-span-2 ${
                            isDark
                              ? 'bg-slate-950/50 border-slate-800'
                              : 'bg-slate-50/70 border-slate-200'
                          }`}
                        >
                          <span className={`font-sans ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                            SGD Classifier
                          </span>
                          <span
                            className={
                              msg.prediction.sgdClassifier.sentiment === 'positive'
                                ? 'text-emerald-500 font-semibold'
                                : 'text-rose-500 font-semibold'
                            }
                          >
                            {msg.prediction.sgdClassifier.sentiment} ({msg.prediction.sgdClassifier.confidence}%)
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {isUser && (
                <div
                  className={`w-7 h-7 rounded-lg border flex items-center justify-center shrink-0 mt-0.5 ${
                    isDark
                      ? 'bg-slate-800 border-slate-700 text-slate-300'
                      : 'bg-slate-200 border-slate-300 text-slate-700'
                  }`}
                >
                  <User className="w-3.5 h-3.5" />
                </div>
              )}
            </div>
          );
        })}

        {isPredictionLoading && (
          <div className="flex items-start gap-3 justify-start animate-pulse">
            <div className="w-7 h-7 rounded-lg bg-indigo-600/20 text-indigo-500 border border-indigo-500/30 flex items-center justify-center shrink-0 mt-0.5">
              <Bot className="w-3.5 h-3.5" />
            </div>
            <div
              className={`p-3.5 rounded-xl border text-xs flex items-center gap-2 ${
                isDark
                  ? 'bg-slate-900 border-slate-800 text-slate-300'
                  : 'bg-white border-slate-200 text-slate-700 shadow-sm'
              }`}
            >
              <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-500" />
              <span>Analyzing review sentiment across ensemble models...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Preset Quick Chips */}
      <div
        className={`border-x px-4 py-2 flex items-center gap-2 overflow-x-auto text-xs shrink-0 transition-colors ${
          isDark
            ? 'bg-slate-900/90 border-slate-800'
            : 'bg-white border-slate-200'
        }`}
      >
        <span className={`text-[11px] shrink-0 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
          Try review:
        </span>
        {PRESET_EXAMPLES.map((example, i) => (
          <button
            key={i}
            onClick={() => handleSend(example)}
            disabled={isPredictionLoading}
            className={`px-2.5 py-1 rounded text-[11px] truncate max-w-[200px] shrink-0 border transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
              isDark
                ? 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700/60'
                : 'bg-slate-100 hover:bg-slate-200 text-slate-700 border-slate-300'
            }`}
            title={example}
          >
            "{example}"
          </button>
        ))}
      </div>

      {/* Chat Input Bar */}
      <div
        className={`border rounded-b-xl p-3 shrink-0 transition-colors ${
          isDark
            ? 'bg-slate-900 border-slate-800'
            : 'bg-white border-slate-200 shadow-sm'
        }`}
      >
        <form
          onSubmit={e => {
            e.preventDefault();
            handleSend();
          }}
          className="flex items-center gap-2"
        >
          <input
            type="text"
            value={inputReview}
            onChange={e => setInputReview(e.target.value)}
            disabled={isPredictionLoading}
            placeholder="Type a movie review (e.g. 'The movie was amazing and I really enjoyed it.')..."
            className={`flex-1 border rounded-lg px-3.5 py-2.5 text-xs transition-colors focus:outline-none focus:border-indigo-500 ${
              isDark
                ? 'bg-slate-950 border-slate-700 text-white placeholder-slate-500'
                : 'bg-slate-50 border-slate-300 text-slate-900 placeholder-slate-400'
            }`}
          />
          <button
            type="submit"
            disabled={!inputReview.trim() || isPredictionLoading}
            className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-semibold rounded-lg flex items-center gap-1.5 transition-colors shrink-0 shadow-sm cursor-pointer"
          >
            {isPredictionLoading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Predicting...</span>
              </>
            ) : (
              <>
                <span>Predict</span>
                <Send className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
};
