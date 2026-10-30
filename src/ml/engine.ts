import { preprocessReview } from './preprocessing';
import { SparseVector, TfidfVectorizer, SerializedTfidf } from './tfidf';
import {
  LinearSVMModel,
  LogisticRegressionModel,
  MultinomialNaiveBayesModel,
  RandomForestModel,
  SGDClassifierModel,
  SoftVotingEnsemble
} from './models';
import { computeMetrics, computeRocCurve, stratifiedSplit } from './evaluation';
import { EvaluationResults, PredictionResult, ReviewItem, RocCurveResults } from '../types';

export interface SerializedEngineState {
  vectorizer: SerializedTfidf;
  lrModel: { weights: number[]; bias: number; isTrained: boolean };
  nbModel: {
    featureLogProbPos: number[];
    featureLogProbNeg: number[];
    logPriorPos: number;
    logPriorNeg: number;
    isTrained: boolean;
  };
  svmModel: {
    weights: number[];
    bias: number;
    plattA: number;
    plattB: number;
    isTrained: boolean;
  };
  rfModel: { trees: any[]; isTrained: boolean };
  sgdModel: { weights: number[]; bias: number; isTrained: boolean };
  isTrained: boolean;
  evaluationResults: EvaluationResults | null;
  trainSetCount: number;
  testSetCount: number;
}

export class SentimentPipelineEngine {
  private vectorizer: TfidfVectorizer = new TfidfVectorizer();
  private lrModel: LogisticRegressionModel = new LogisticRegressionModel();
  private nbModel: MultinomialNaiveBayesModel = new MultinomialNaiveBayesModel();
  private svmModel: LinearSVMModel = new LinearSVMModel();
  private rfModel: RandomForestModel = new RandomForestModel();
  private sgdModel: SGDClassifierModel = new SGDClassifierModel();
  private ensemble: SoftVotingEnsemble = new SoftVotingEnsemble();

  private isTrained: boolean = false;
  private evaluationResults: EvaluationResults | null = null;
  private trainSetCount: number = 0;
  private testSetCount: number = 0;

  /**
   * Trains the full ML pipeline on the provided dataset.
   * Runs: Preprocessing -> TF-IDF -> Stratified Split -> 5 Models -> Soft Voting -> Evaluation
   */
  public async train(
    dataset: ReviewItem[],
    onProgress?: (step: string, percentage: number) => void
  ): Promise<EvaluationResults> {
    onProgress?.('Text Preprocessing', 15);
    await new Promise(r => setTimeout(r, 40));

    // Step 1: Preprocess all reviews into unigram/bigram tokens
    const preprocessedDocs: { item: ReviewItem; tokens: string[] }[] = [];
    const chunkSize = 250;
    for (let i = 0; i < dataset.length; i += chunkSize) {
      const slice = dataset.slice(i, i + chunkSize);
      for (const item of slice) {
        preprocessedDocs.push({
          item,
          tokens: preprocessReview(item.review)
        });
      }
      if (dataset.length > 250) {
        await new Promise(r => setTimeout(r, 0));
      }
    }

    onProgress?.('Stratified Train/Test Split (80/20)', 30);
    await new Promise(r => setTimeout(r, 40));

    // Step 2: Stratified 80/20 split
    const { train, test } = stratifiedSplit(dataset, 0.8);
    this.trainSetCount = train.length;
    this.testSetCount = test.length;

    const trainIds = new Set(train.map(t => t.id));
    const trainDocs = preprocessedDocs.filter(d => trainIds.has(d.item.id));
    const testDocs = preprocessedDocs.filter(d => !trainIds.has(d.item.id));

    onProgress?.('TF-IDF Vectorization', 45);
    await new Promise(r => setTimeout(r, 20));

    // Step 3: Fit TF-IDF on training set only (prevent data leakage)
    this.vectorizer = new TfidfVectorizer({ minDocFreq: dataset.length < 500 ? 1 : 2, maxFeatures: 3000 });
    const trainTokens = trainDocs.map(d => d.tokens);
    const X_train: SparseVector[] = this.vectorizer.fitTransform(trainTokens);
    const y_train: number[] = trainDocs.map(d => (d.item.sentiment === 'positive' ? 1 : 0));
    const vocabSize = this.vectorizer.getVocabularySize();

    // Transform test set using the fitted vocabulary
    const testTokens = testDocs.map(d => d.tokens);
    const X_test: SparseVector[] = this.vectorizer.transform(testTokens);
    const y_test_truth = testDocs.map(d => d.item.sentiment);

    // Step 4: Train all 5 classifiers with event-loop yields to keep UI responsive
    onProgress?.('Training Logistic Regression...', 55);
    await new Promise(r => setTimeout(r, 15));
    this.lrModel.train(X_train, y_train, vocabSize);

    onProgress?.('Training Naive Bayes...', 65);
    await new Promise(r => setTimeout(r, 15));
    this.nbModel.train(X_train, y_train, vocabSize);

    onProgress?.('Training Linear SVM...', 75);
    await new Promise(r => setTimeout(r, 15));
    this.svmModel.train(X_train, y_train, vocabSize);

    onProgress?.('Training Random Forest...', 85);
    await new Promise(r => setTimeout(r, 15));
    this.rfModel.train(X_train, y_train, vocabSize);

    onProgress?.('Training SGD Classifier...', 92);
    await new Promise(r => setTimeout(r, 15));
    this.sgdModel.train(X_train, y_train, vocabSize);

    onProgress?.('Updating ensemble & test metrics...', 96);
    await new Promise(r => setTimeout(r, 15));

    // Step 5: Evaluate all 5 models and ensemble on test set
    const predsLR: string[] = [];
    const predsNB: string[] = [];
    const predsSVM: string[] = [];
    const predsRF: string[] = [];
    const predsSGD: string[] = [];
    const predsEnsemble: string[] = [];

    const scoresLR: number[] = [];
    const scoresNB: number[] = [];
    const scoresSVM: number[] = [];
    const scoresRF: number[] = [];
    const scoresSGD: number[] = [];
    const scoresEnsemble: number[] = [];

    for (let i = 0; i < X_test.length; i++) {
      const vec = X_test[i];
      const pLR = this.lrModel.predictProb(vec);
      const pNB = this.nbModel.predictProb(vec);
      const pSVM = this.svmModel.predictProb(vec);
      const pRF = this.rfModel.predictProb(vec);
      const pSGD = this.sgdModel.predictProb(vec);

      scoresLR.push(pLR);
      scoresNB.push(pNB);
      scoresSVM.push(pSVM);
      scoresRF.push(pRF);
      scoresSGD.push(pSGD);

      predsLR.push(pLR >= 0.5 ? 'positive' : 'negative');
      predsNB.push(pNB >= 0.5 ? 'positive' : 'negative');
      predsSVM.push(pSVM >= 0.5 ? 'positive' : 'negative');
      predsRF.push(pRF >= 0.5 ? 'positive' : 'negative');
      predsSGD.push(pSGD >= 0.5 ? 'positive' : 'negative');

      const ensPred = this.ensemble.predictEnsemble(pLR, pNB, pSVM, pRF, pSGD);
      predsEnsemble.push(ensPred.sentiment);
      scoresEnsemble.push(ensPred.positiveProbability);
    }

    const metricLR = computeMetrics('Logistic Regression', y_test_truth, predsLR as any, scoresLR);
    const metricNB = computeMetrics('Multinomial Naive Bayes', y_test_truth, predsNB as any, scoresNB);
    const metricSVM = computeMetrics('Linear SVM', y_test_truth, predsSVM as any, scoresSVM);
    const metricRF = computeMetrics('Random Forest', y_test_truth, predsRF as any, scoresRF);
    const metricSGD = computeMetrics('SGD Classifier', y_test_truth, predsSGD as any, scoresSGD);
    const metricEnsemble = computeMetrics('Soft Voting Ensemble', y_test_truth, predsEnsemble as any, scoresEnsemble);

    // Compute ROC curves and AUCs for all models and the ensemble
    const rocLR = computeRocCurve(y_test_truth, scoresLR);
    const rocNB = computeRocCurve(y_test_truth, scoresNB);
    const rocSVM = computeRocCurve(y_test_truth, scoresSVM);
    const rocRF = computeRocCurve(y_test_truth, scoresRF);
    const rocSGD = computeRocCurve(y_test_truth, scoresSGD);
    const rocEns = computeRocCurve(y_test_truth, scoresEnsemble);

    let posCount = 0;
    let negCount = 0;
    for (const label of y_test_truth) {
      if (label === 'positive') posCount++;
      else negCount++;
    }

    const hasValidRoc = !!(rocLR && rocNB && rocSVM && rocRF && rocSGD && rocEns && posCount > 0 && negCount > 0);

    const rocCurveResults: RocCurveResults = {
      hasValidData: hasValidRoc,
      errorMessage: hasValidRoc
        ? undefined
        : 'Insufficient positive or negative test-set examples to generate ROC curve.',
      testSetCount: y_test_truth.length,
      positiveCount: posCount,
      negativeCount: negCount,
      models: {
        logisticRegression: rocLR ? { modelName: 'Logistic Regression', auc: rocLR.auc, points: rocLR.points } : undefined,
        naiveBayes: rocNB ? { modelName: 'Multinomial Naive Bayes', auc: rocNB.auc, points: rocNB.points } : undefined,
        linearSVM: rocSVM ? { modelName: 'Linear SVM', auc: rocSVM.auc, points: rocSVM.points } : undefined,
        randomForest: rocRF ? { modelName: 'Random Forest', auc: rocRF.auc, points: rocRF.points } : undefined,
        sgdClassifier: rocSGD ? { modelName: 'SGD Classifier', auc: rocSGD.auc, points: rocSGD.points } : undefined,
        ensemble: rocEns ? { modelName: 'Soft Voting Ensemble', auc: rocEns.auc, points: rocEns.points } : undefined
      }
    };

    const allMetrics = [metricLR, metricNB, metricSVM, metricRF, metricSGD, metricEnsemble];
    let bestModel = metricEnsemble;
    for (const m of allMetrics) {
      if (m.accuracy > bestModel.accuracy) {
        bestModel = m;
      }
    }

    this.evaluationResults = {
      models: {
        logisticRegression: metricLR,
        naiveBayes: metricNB,
        linearSVM: metricSVM,
        randomForest: metricRF,
        sgdClassifier: metricSGD,
        ensemble: metricEnsemble
      },
      bestModelName: bestModel.modelName,
      bestAccuracy: bestModel.accuracy,
      testSetSize: this.testSetCount,
      trainSetSize: this.trainSetCount,
      rocCurve: rocCurveResults
    };

    this.isTrained = true;
    onProgress?.('Training Complete', 100);

    return this.evaluationResults;
  }

  /**
   * Predicts sentiment for an arbitrary review text using all 5 models and soft voting ensemble.
   */
  public predict(reviewText: string): PredictionResult {
    const tokens = preprocessReview(reviewText);
    const vec = this.vectorizer.transformDoc(tokens);

    const lrPred = this.lrModel.predict(vec);
    const nbPred = this.nbModel.predict(vec);
    const svmPred = this.svmModel.predict(vec);
    const rfPred = this.rfModel.predict(vec);
    const sgdPred = this.sgdModel.predict(vec);

    const ensPred = this.ensemble.predictEnsemble(
      lrPred.positiveProbability,
      nbPred.positiveProbability,
      svmPred.positiveProbability,
      rfPred.positiveProbability,
      sgdPred.positiveProbability
    );

    return {
      logisticRegression: lrPred,
      naiveBayes: nbPred,
      linearSVM: svmPred,
      randomForest: rfPred,
      sgdClassifier: sgdPred,
      ensemble: ensPred
    };
  }

  public getEvaluationResults(): EvaluationResults | null {
    return this.evaluationResults;
  }

  public isModelTrained(): boolean {
    return this.isTrained;
  }

  public getStatus() {
    return {
      isTrained: this.isTrained,
      trainSetCount: this.trainSetCount,
      testSetCount: this.testSetCount,
      vocabSize: this.vectorizer.getVocabularySize()
    };
  }

  public serialize(): SerializedEngineState {
    return {
      vectorizer: this.vectorizer.serialize(),
      lrModel: this.lrModel.serialize(),
      nbModel: this.nbModel.serialize(),
      svmModel: this.svmModel.serialize(),
      rfModel: this.rfModel.serialize(),
      sgdModel: this.sgdModel.serialize(),
      isTrained: this.isTrained,
      evaluationResults: this.evaluationResults,
      trainSetCount: this.trainSetCount,
      testSetCount: this.testSetCount
    };
  }

  public loadSerializedState(state: SerializedEngineState): void {
    if (!state) return;
    if (state.vectorizer) this.vectorizer.loadState(state.vectorizer);
    if (state.lrModel) this.lrModel.loadState(state.lrModel);
    if (state.nbModel) this.nbModel.loadState(state.nbModel);
    if (state.svmModel) this.svmModel.loadState(state.svmModel);
    if (state.rfModel) this.rfModel.loadState(state.rfModel);
    if (state.sgdModel) this.sgdModel.loadState(state.sgdModel);
    this.isTrained = !!state.isTrained;
    this.evaluationResults = state.evaluationResults || null;
    this.trainSetCount = state.trainSetCount || 0;
    this.testSetCount = state.testSetCount || 0;
  }
}

// Global pipeline engine instance
export const sentimentEngine = new SentimentPipelineEngine();
