import { SparseVector } from './tfidf';
import { ModelPrediction, Sentiment } from '../types';

/**
 * Utility: Compute dot product between sparse vector and dense weights array.
 */
function sparseDotProduct(vec: SparseVector, weights: Float64Array): number {
  let sum = 0;
  const indices = vec.indices;
  const values = vec.values;
  const len = indices.length;
  for (let i = 0; i < len; i++) {
    sum += weights[indices[i]] * values[i];
  }
  return sum;
}

/**
 * Utility: Sigmoid function with overflow clamping.
 */
function sigmoid(z: number): number {
  if (z > 35) return 1.0;
  if (z < -35) return 0.0;
  return 1.0 / (1.0 + Math.exp(-z));
}

// ==========================================
// 1. LOGISTIC REGRESSION (L2 Regularized SGD)
// ==========================================
export class LogisticRegressionModel {
  private weights: Float64Array = new Float64Array(0);
  private bias: number = 0;
  private isTrained: boolean = false;

  public train(X: SparseVector[], y: number[], vocabSize: number, epochs: number = 10, lr: number = 0.25, l2: number = 0.0005): void {
    this.weights = new Float64Array(vocabSize);
    this.bias = 0;
    const n = X.length;
    if (n === 0) return;

    for (let epoch = 0; epoch < epochs; epoch++) {
      const currentLr = lr / (1 + 0.02 * epoch);
      for (let i = 0; i < n; i++) {
        const x_i = X[i];
        const y_i = y[i];
        const z = sparseDotProduct(x_i, this.weights) + this.bias;
        const p = sigmoid(z);
        const error = p - y_i;

        // Gradient step with L2 penalty on active features
        const indices = x_i.indices;
        const values = x_i.values;
        for (let j = 0; j < indices.length; j++) {
          const idx = indices[j];
          const grad = error * values[j] + l2 * this.weights[idx];
          this.weights[idx] -= currentLr * grad;
        }
        this.bias -= currentLr * error * 0.05;
      }
    }
    this.isTrained = true;
  }

  public predictProb(x: SparseVector): number {
    if (!this.isTrained) return 0.5;
    const z = sparseDotProduct(x, this.weights) + this.bias;
    return sigmoid(z);
  }

  public predict(x: SparseVector): ModelPrediction {
    const prob = this.predictProb(x);
    const sentiment: Sentiment = prob >= 0.5 ? 'positive' : 'negative';
    const confidence = Math.round(Math.max(prob, 1 - prob) * 100);
    return { sentiment, confidence, positiveProbability: prob };
  }

  public serialize() {
    return {
      weights: Array.from(this.weights),
      bias: this.bias,
      isTrained: this.isTrained
    };
  }

  public loadState(state: { weights: number[]; bias: number; isTrained: boolean }) {
    if (!state) return;
    this.weights = new Float64Array(state.weights || []);
    this.bias = state.bias ?? 0;
    this.isTrained = !!state.isTrained;
  }
}

// ==========================================
// 2. MULTINOMIAL NAIVE BAYES (Laplace Smoothing)
// ==========================================
export class MultinomialNaiveBayesModel {
  private featureLogProbPos: Float64Array = new Float64Array(0);
  private featureLogProbNeg: Float64Array = new Float64Array(0);
  private logPriorPos: number = 0;
  private logPriorNeg: number = 0;
  private isTrained: boolean = false;

  public train(X: SparseVector[], y: number[], vocabSize: number, alpha: number = 1.0): void {
    const n = X.length;
    if (n === 0) return;

    let countPos = 0;
    let countNeg = 0;

    const featureSumPos = new Float64Array(vocabSize);
    const featureSumNeg = new Float64Array(vocabSize);
    let totalWordSumPos = 0;
    let totalWordSumNeg = 0;

    for (let i = 0; i < n; i++) {
      const isPos = y[i] === 1;
      if (isPos) countPos++;
      else countNeg++;

      const indices = X[i].indices;
      const values = X[i].values;
      for (let j = 0; j < indices.length; j++) {
        const idx = indices[j];
        const val = values[j];
        if (isPos) {
          featureSumPos[idx] += val;
          totalWordSumPos += val;
        } else {
          featureSumNeg[idx] += val;
          totalWordSumNeg += val;
        }
      }
    }

    this.logPriorPos = Math.log((countPos + alpha) / (n + 2 * alpha));
    this.logPriorNeg = Math.log((countNeg + alpha) / (n + 2 * alpha));

    this.featureLogProbPos = new Float64Array(vocabSize);
    this.featureLogProbNeg = new Float64Array(vocabSize);

    const denomPos = totalWordSumPos + alpha * vocabSize;
    const denomNeg = totalWordSumNeg + alpha * vocabSize;

    for (let i = 0; i < vocabSize; i++) {
      this.featureLogProbPos[i] = Math.log((featureSumPos[i] + alpha) / denomPos);
      this.featureLogProbNeg[i] = Math.log((featureSumNeg[i] + alpha) / denomNeg);
    }

    this.isTrained = true;
  }

  public predictProb(x: SparseVector): number {
    if (!this.isTrained) return 0.5;

    let logLikelihoodPos = this.logPriorPos;
    let logLikelihoodNeg = this.logPriorNeg;

    const indices = x.indices;
    const values = x.values;
    for (let j = 0; j < indices.length; j++) {
      const idx = indices[j];
      const val = values[j];
      logLikelihoodPos += val * this.featureLogProbPos[idx];
      logLikelihoodNeg += val * this.featureLogProbNeg[idx];
    }

    // Softmax via log-sum-exp: P(pos) = 1 / (1 + exp(logNeg - logPos))
    const diff = logLikelihoodNeg - logLikelihoodPos;
    if (diff > 35) return 0.0;
    if (diff < -35) return 1.0;
    return 1.0 / (1.0 + Math.exp(diff));
  }

  public predict(x: SparseVector): ModelPrediction {
    const prob = this.predictProb(x);
    const sentiment: Sentiment = prob >= 0.5 ? 'positive' : 'negative';
    const confidence = Math.round(Math.max(prob, 1 - prob) * 100);
    return { sentiment, confidence, positiveProbability: prob };
  }

  public serialize() {
    return {
      featureLogProbPos: Array.from(this.featureLogProbPos),
      featureLogProbNeg: Array.from(this.featureLogProbNeg),
      logPriorPos: this.logPriorPos,
      logPriorNeg: this.logPriorNeg,
      isTrained: this.isTrained
    };
  }

  public loadState(state: {
    featureLogProbPos: number[];
    featureLogProbNeg: number[];
    logPriorPos: number;
    logPriorNeg: number;
    isTrained: boolean;
  }) {
    if (!state) return;
    this.featureLogProbPos = new Float64Array(state.featureLogProbPos || []);
    this.featureLogProbNeg = new Float64Array(state.featureLogProbNeg || []);
    this.logPriorPos = state.logPriorPos ?? 0;
    this.logPriorNeg = state.logPriorNeg ?? 0;
    this.isTrained = !!state.isTrained;
  }
}

// ==========================================
// 3. LINEAR SVM (Hinge Loss + Platt Scaling)
// ==========================================
export class LinearSVMModel {
  private weights: Float64Array = new Float64Array(0);
  private bias: number = 0;
  private plattA: number = 1.0;
  private plattB: number = 0.0;
  private isTrained: boolean = false;

  public train(X: SparseVector[], y: number[], vocabSize: number, epochs: number = 10, lambdaParam: number = 0.001): void {
    this.weights = new Float64Array(vocabSize);
    this.bias = 0;
    const n = X.length;
    if (n === 0) return;

    // Convert labels from {0, 1} to {-1, +1} for SVM formulation
    const labels = y.map(v => (v === 1 ? 1 : -1));

    let t = 0;
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < n; i++) {
        t++;
        const eta = 1.0 / (lambdaParam * t + 10.0);
        const x_i = X[i];
        const y_i = labels[i];

        const margin = y_i * (sparseDotProduct(x_i, this.weights) + this.bias);
        if (margin < 1.0) {
          // Misclassified or within margin: subgradient update
          const indices = x_i.indices;
          const values = x_i.values;
          for (let j = 0; j < indices.length; j++) {
            const idx = indices[j];
            this.weights[idx] = (1.0 - eta * lambdaParam) * this.weights[idx] + eta * y_i * values[j];
          }
          this.bias += eta * y_i * 0.1;
        } else {
          // Correctly classified outside margin: shrink weights
          const indices = x_i.indices;
          for (let j = 0; j < indices.length; j++) {
            const idx = indices[j];
            this.weights[idx] *= 1.0 - eta * lambdaParam;
          }
        }
      }
    }

    // Platt Scaling calibration on training margins
    this.fitPlattCalibration(X, y);
    this.isTrained = true;
  }

  private fitPlattCalibration(X: SparseVector[], y: number[]): void {
    let sumMarginPos = 0;
    let sumMarginNeg = 0;
    let countPos = 0;
    let countNeg = 0;

    for (let i = 0; i < X.length; i++) {
      const margin = sparseDotProduct(X[i], this.weights) + this.bias;
      if (y[i] === 1) {
        sumMarginPos += margin;
        countPos++;
      } else {
        sumMarginNeg += margin;
        countNeg++;
      }
    }

    const avgPos = countPos > 0 ? sumMarginPos / countPos : 1;
    const avgNeg = countNeg > 0 ? sumMarginNeg / countNeg : -1;
    const spread = Math.max(0.2, avgPos - avgNeg);

    this.plattA = 2.4 / spread;
    this.plattB = -this.plattA * ((avgPos + avgNeg) / 2);
  }

  public decisionFunction(x: SparseVector): number {
    return sparseDotProduct(x, this.weights) + this.bias;
  }

  public predictProb(x: SparseVector): number {
    if (!this.isTrained) return 0.5;
    const f = this.decisionFunction(x);
    // Platt scaling formula: 1 / (1 + exp(-(A * f + B)))
    return sigmoid(this.plattA * f + this.plattB);
  }

  public predict(x: SparseVector): ModelPrediction {
    const prob = this.predictProb(x);
    const sentiment: Sentiment = prob >= 0.5 ? 'positive' : 'negative';
    const confidence = Math.round(Math.max(prob, 1 - prob) * 100);
    return { sentiment, confidence, positiveProbability: prob };
  }

  public serialize() {
    return {
      weights: Array.from(this.weights),
      bias: this.bias,
      plattA: this.plattA,
      plattB: this.plattB,
      isTrained: this.isTrained
    };
  }

  public loadState(state: {
    weights: number[];
    bias: number;
    plattA: number;
    plattB: number;
    isTrained: boolean;
  }) {
    if (!state) return;
    this.weights = new Float64Array(state.weights || []);
    this.bias = state.bias ?? 0;
    this.plattA = state.plattA ?? 1.0;
    this.plattB = state.plattB ?? 0.0;
    this.isTrained = !!state.isTrained;
  }
}

// ==========================================
// 4. RANDOM FOREST (Bagged Decision Stumps / Trees)
// ==========================================
interface TreeNode {
  isLeaf: boolean;
  featureIndex?: number;
  threshold?: number;
  probPos: number;
  left?: TreeNode;
  right?: TreeNode;
}

export class RandomForestModel {
  private trees: TreeNode[] = [];
  private numTrees: number = 12;
  private maxDepth: number = 3;
  private isTrained: boolean = false;

  public train(X: SparseVector[], y: number[], vocabSize: number): void {
    const n = X.length;
    if (n === 0) return;

    this.trees = [];
    const numFeaturesSample = Math.max(8, Math.min(24, Math.floor(Math.sqrt(vocabSize))));
    const sampleSize = Math.min(n, 120);

    // Create random trees with bootstrap sampling
    for (let t = 0; t < this.numTrees; t++) {
      const sampleIndices: number[] = new Array(sampleSize);
      for (let i = 0; i < sampleSize; i++) {
        sampleIndices[i] = Math.floor(Math.random() * n);
      }

      const tree = this.buildNode(X, y, sampleIndices, vocabSize, numFeaturesSample, 0);
      this.trees.push(tree);
    }

    this.isTrained = true;
  }

  private buildNode(
    X: SparseVector[],
    y: number[],
    indices: number[],
    vocabSize: number,
    numFeaturesSample: number,
    depth: number
  ): TreeNode {
    // Count positive and negative in this node
    let posCount = 0;
    for (const idx of indices) {
      if (y[idx] === 1) posCount++;
    }
    const nodeSize = indices.length;
    const probPos = (posCount + 1) / (nodeSize + 2); // Laplace smoothed node prob

    // Base cases: max depth, pure node, or small node
    if (depth >= this.maxDepth || nodeSize <= 4 || posCount === 0 || posCount === nodeSize) {
      return { isLeaf: true, probPos };
    }

    // Select random subset of candidate features
    const candidateFeatures: number[] = [];
    for (let f = 0; f < numFeaturesSample; f++) {
      candidateFeatures.push(Math.floor(Math.random() * vocabSize));
    }

    let bestGini = 1.0;
    let bestFeature = -1;
    let bestThreshold = 0.05;
    let bestLeftIndices: number[] = [];
    let bestRightIndices: number[] = [];

    // Current node Gini impurity
    const p1 = posCount / nodeSize;
    const p0 = 1 - p1;
    const currentGini = 1 - (p1 * p1 + p0 * p0);

    for (const feat of candidateFeatures) {
      // Check binary presence/threshold split: value > 0.05
      const threshold = 0.05;
      const left: number[] = [];
      const right: number[] = [];

      for (const idx of indices) {
        const val = this.getFeatureValue(X[idx], feat);
        if (val <= threshold) {
          left.push(idx);
        } else {
          right.push(idx);
        }
      }

      if (left.length === 0 || right.length === 0) continue;

      let leftPos = 0;
      for (const idx of left) if (y[idx] === 1) leftPos++;
      const lp1 = leftPos / left.length;
      const leftGini = 1 - (lp1 * lp1 + (1 - lp1) * (1 - lp1));

      let rightPos = 0;
      for (const idx of right) if (y[idx] === 1) rightPos++;
      const rp1 = rightPos / right.length;
      const rightGini = 1 - (rp1 * rp1 + (1 - rp1) * (1 - rp1));

      const weightedGini = (left.length / nodeSize) * leftGini + (right.length / nodeSize) * rightGini;

      if (weightedGini < bestGini) {
        bestGini = weightedGini;
        bestFeature = feat;
        bestThreshold = threshold;
        bestLeftIndices = left;
        bestRightIndices = right;
      }
    }

    // If no informative split found, make leaf
    if (bestFeature === -1 || bestGini >= currentGini) {
      return { isLeaf: true, probPos };
    }

    return {
      isLeaf: false,
      featureIndex: bestFeature,
      threshold: bestThreshold,
      probPos,
      left: this.buildNode(X, y, bestLeftIndices, vocabSize, numFeaturesSample, depth + 1),
      right: this.buildNode(X, y, bestRightIndices, vocabSize, numFeaturesSample, depth + 1)
    };
  }

  private getFeatureValue(x: SparseVector, featIndex: number): number {
    const indices = x.indices;
    let low = 0;
    let high = indices.length - 1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      const midVal = indices[mid];
      if (midVal === featIndex) return x.values[mid];
      if (midVal < featIndex) low = mid + 1;
      else high = mid - 1;
    }
    return 0;
  }

  private predictTreeProb(node: TreeNode, x: SparseVector): number {
    if (node.isLeaf || node.featureIndex === undefined || node.threshold === undefined) {
      return node.probPos;
    }
    const val = this.getFeatureValue(x, node.featureIndex);
    if (val <= node.threshold) {
      return node.left ? this.predictTreeProb(node.left, x) : node.probPos;
    } else {
      return node.right ? this.predictTreeProb(node.right, x) : node.probPos;
    }
  }

  public predictProb(x: SparseVector): number {
    if (!this.isTrained || this.trees.length === 0) return 0.5;
    let totalProb = 0;
    for (const tree of this.trees) {
      totalProb += this.predictTreeProb(tree, x);
    }
    return totalProb / this.trees.length;
  }

  public predict(x: SparseVector): ModelPrediction {
    const prob = this.predictProb(x);
    const sentiment: Sentiment = prob >= 0.5 ? 'positive' : 'negative';
    const confidence = Math.round(Math.max(prob, 1 - prob) * 100);
    return { sentiment, confidence, positiveProbability: prob };
  }

  public serialize() {
    return {
      trees: this.trees,
      isTrained: this.isTrained
    };
  }

  public loadState(state: { trees: TreeNode[]; isTrained: boolean }) {
    if (!state) return;
    this.trees = Array.isArray(state.trees) ? state.trees : [];
    this.isTrained = !!state.isTrained;
  }
}

// ==========================================
// 5. SGD CLASSIFIER (Modified Huber Loss)
// ==========================================
export class SGDClassifierModel {
  private weights: Float64Array = new Float64Array(0);
  private bias: number = 0;
  private isTrained: boolean = false;

  public train(X: SparseVector[], y: number[], vocabSize: number, epochs: number = 10, alpha: number = 0.0001): void {
    this.weights = new Float64Array(vocabSize);
    this.bias = 0;
    const n = X.length;
    if (n === 0) return;

    // Convert labels from {0, 1} to {-1, +1}
    const labels = y.map(v => (v === 1 ? 1 : -1));
    let t = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < n; i++) {
        t++;
        const eta = 1.0 / (alpha * t + 1.0);
        const x_i = X[i];
        const y_i = labels[i];

        const f_x = sparseDotProduct(x_i, this.weights) + this.bias;
        const z = y_i * f_x;

        // Modified Huber Loss gradient:
        // L(y, f) = max(0, 1 - z)^2 for z >= -1, else -4z
        let lossGrad = 0;
        if (z >= 1.0) {
          lossGrad = 0;
        } else if (z >= -1.0) {
          lossGrad = -2.0 * (1.0 - z) * y_i;
        } else {
          lossGrad = -4.0 * y_i;
        }

        const indices = x_i.indices;
        const values = x_i.values;
        for (let j = 0; j < indices.length; j++) {
          const idx = indices[j];
          this.weights[idx] = (1.0 - eta * alpha) * this.weights[idx] - eta * lossGrad * values[j];
        }
        this.bias -= eta * lossGrad * 0.05;
      }
    }
    this.isTrained = true;
  }

  public predictProb(x: SparseVector): number {
    if (!this.isTrained) return 0.5;
    const f_x = sparseDotProduct(x, this.weights) + this.bias;

    // Modified Huber calibrated probability:
    // P(y=1) = max(0, min(1, (f(x) + 1) / 2))
    const rawProb = (f_x + 1.0) / 2.0;
    return Math.max(0.01, Math.min(0.99, rawProb));
  }

  public predict(x: SparseVector): ModelPrediction {
    const prob = this.predictProb(x);
    const sentiment: Sentiment = prob >= 0.5 ? 'positive' : 'negative';
    const confidence = Math.round(Math.max(prob, 1 - prob) * 100);
    return { sentiment, confidence, positiveProbability: prob };
  }

  public serialize() {
    return {
      weights: Array.from(this.weights),
      bias: this.bias,
      isTrained: this.isTrained
    };
  }

  public loadState(state: { weights: number[]; bias: number; isTrained: boolean }) {
    if (!state) return;
    this.weights = new Float64Array(state.weights || []);
    this.bias = state.bias ?? 0;
    this.isTrained = !!state.isTrained;
  }
}

// ==========================================
// 6. SOFT VOTING ENSEMBLE
// ==========================================
export class SoftVotingEnsemble {
  public predictEnsemble(
    pLR: number,
    pNB: number,
    pSVM: number,
    pRF: number,
    pSGD: number
  ): ModelPrediction {
    // Soft voting computes the arithmetic mean of positive class probabilities
    const ensembleProb = (pLR + pNB + pSVM + pRF + pSGD) / 5.0;
    const sentiment: Sentiment = ensembleProb >= 0.5 ? 'positive' : 'negative';
    const confidence = Math.round(Math.max(ensembleProb, 1.0 - ensembleProb) * 100);

    return {
      sentiment,
      confidence,
      positiveProbability: ensembleProb
    };
  }
}
