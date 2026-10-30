export type Sentiment = 'positive' | 'negative';
export type Theme = 'light' | 'dark';

export interface ReviewItem {
  id: string;
  review: string;
  sentiment: Sentiment;
  cleanedText?: string;
}

export interface UploadedDatasetRecord {
  id: string;
  name: string;
  examplesCount: number;
  status: 'Trained' | 'Pending Training' | 'Training';
  uploadedRows?: number;
  duplicateRows?: number;
}

export interface DatasetStats {
  total: number;
  positive: number;
  negative: number;
  missingValues: number;
  duplicateRecords: number;
  trainCount: number;
  testCount: number;
  sourceName: string;
  isDemo: boolean;
  datasetsCount: number;
  modelVersion: number;
  lastTrainedTime: string | null;
  uploadedDatasets: UploadedDatasetRecord[];
  lastBatchAddedCount?: number;
  lastBatchDuplicatesRemoved?: number;
}

export interface ModelPrediction {
  sentiment: Sentiment;
  confidence: number; // 0 to 100
  positiveProbability: number; // 0.0 to 1.0
}

export interface PredictionResult {
  logisticRegression: ModelPrediction;
  naiveBayes: ModelPrediction;
  linearSVM: ModelPrediction;
  randomForest: ModelPrediction;
  sgdClassifier: ModelPrediction;
  ensemble: ModelPrediction;
}

export interface ConfusionMatrix {
  truePositive: number;
  trueNegative: number;
  falsePositive: number;
  falseNegative: number;
  total: number;
}

export interface ModelMetrics {
  modelName: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc?: number;
  confusionMatrix?: ConfusionMatrix;
}

export interface RocPoint {
  fpr: number; // 0.0 to 1.0
  tpr: number; // 0.0 to 1.0
  threshold?: number;
}

export interface ModelRocData {
  modelName: string;
  auc: number; // e.g. 0.954
  points: RocPoint[];
}

export interface RocCurveResults {
  models: {
    logisticRegression?: ModelRocData;
    naiveBayes?: ModelRocData;
    linearSVM?: ModelRocData;
    randomForest?: ModelRocData;
    sgdClassifier?: ModelRocData;
    ensemble?: ModelRocData;
  };
  hasValidData: boolean;
  errorMessage?: string;
  testSetCount?: number;
  positiveCount?: number;
  negativeCount?: number;
}

export interface EvaluationResults {
  models: {
    logisticRegression: ModelMetrics;
    naiveBayes: ModelMetrics;
    linearSVM: ModelMetrics;
    randomForest: ModelMetrics;
    sgdClassifier: ModelMetrics;
    ensemble: ModelMetrics;
  };
  bestModelName: string;
  bestAccuracy: number;
  testSetSize: number;
  trainSetSize: number;
  vocabSize?: number;
  rocCurve?: RocCurveResults;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: string;
  prediction?: PredictionResult;
  modelVersion?: number;
}

export interface BatchReviewItem {
  id: string;
  review: string;
  prediction?: PredictionResult;
}

export type ActiveTab = 
  | 'dashboard'
  | 'chatbot'
  | 'dataset'
  | 'train'
  | 'evaluation'
  | 'visualization';
