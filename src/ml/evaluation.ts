import { ConfusionMatrix, ModelMetrics, ReviewItem, RocPoint, Sentiment } from '../types';

/**
 * Performs a stratified 80/20 train-test split to preserve sentiment class proportions.
 */
export function stratifiedSplit(
  items: ReviewItem[],
  trainRatio: number = 0.8
): { train: ReviewItem[]; test: ReviewItem[] } {
  const positives = items.filter(item => item.sentiment === 'positive');
  const negatives = items.filter(item => item.sentiment === 'negative');

  // Seeded linear congruential pseudo-random generator for reproducible, stable splits
  const rng = (() => {
    let s = 1337;
    return () => {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    };
  })();

  const shuffle = <T>(arr: T[]): T[] => {
    const copy = [...arr];
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  };

  const shuffledPos = shuffle(positives);
  const shuffledNeg = shuffle(negatives);

  const trainPosCount = Math.floor(shuffledPos.length * trainRatio);
  const trainNegCount = Math.floor(shuffledNeg.length * trainRatio);

  const train: ReviewItem[] = [
    ...shuffledPos.slice(0, trainPosCount),
    ...shuffledNeg.slice(0, trainNegCount)
  ];

  const test: ReviewItem[] = [
    ...shuffledPos.slice(trainPosCount),
    ...shuffledNeg.slice(trainNegCount)
  ];

  return {
    train: shuffle(train),
    test: shuffle(test)
  };
}

/**
 * Computes ROC curve (FPR, TPR points) and Area Under the Curve (AUC)
 * using the trapezoidal rule on the positive class predictions against ground truth.
 */
export function computeRocCurve(
  groundTruth: Sentiment[],
  positiveProbabilities: number[]
): { points: RocPoint[]; auc: number } | null {
  if (!groundTruth || !positiveProbabilities || groundTruth.length !== positiveProbabilities.length) {
    return null;
  }

  const n = groundTruth.length;
  if (n === 0) return null;

  let totalPos = 0;
  let totalNeg = 0;
  for (let i = 0; i < n; i++) {
    if (groundTruth[i] === 'positive') totalPos++;
    else if (groundTruth[i] === 'negative') totalNeg++;
  }

  // Gracefully handle insufficient class data (must have both classes in test set)
  if (totalPos === 0 || totalNeg === 0) {
    return null;
  }

  // Pair scores with binary truth (1 for positive, 0 for negative)
  const pairs: Array<{ isPos: boolean; score: number }> = new Array(n);
  for (let i = 0; i < n; i++) {
    pairs[i] = {
      isPos: groundTruth[i] === 'positive',
      score: positiveProbabilities[i]
    };
  }

  // Sort descending by positive prediction score
  pairs.sort((a, b) => b.score - a.score);

  const points: RocPoint[] = [{ fpr: 0, tpr: 0, threshold: 1.0 }];

  let tp = 0;
  let fp = 0;

  for (let i = 0; i < n; i++) {
    if (pairs[i].isPos) {
      tp++;
    } else {
      fp++;
    }

    // Only add a point when threshold changes or at the end of array
    if (i === n - 1 || pairs[i].score !== pairs[i + 1].score) {
      const fpr = fp / totalNeg;
      const tpr = tp / totalPos;
      points.push({
        fpr: Number(fpr.toFixed(4)),
        tpr: Number(tpr.toFixed(4)),
        threshold: Number(pairs[i].score.toFixed(4))
      });
    }
  }

  // Ensure curve ends at (1, 1)
  const lastPoint = points[points.length - 1];
  if (lastPoint.fpr < 1 || lastPoint.tpr < 1) {
    points.push({ fpr: 1, tpr: 1, threshold: 0 });
  }

  // Calculate AUC via trapezoidal integration
  let auc = 0;
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    auc += ((curr.tpr + prev.tpr) / 2) * (curr.fpr - prev.fpr);
  }

  // Clamp AUC between 0 and 1
  auc = Math.max(0, Math.min(1, auc));

  return {
    points,
    auc: Number(auc.toFixed(4))
  };
}

/**
 * Computes confusion matrix and standard classification metrics on ground truth vs predictions.
 */
export function computeMetrics(
  modelName: string,
  groundTruth: Sentiment[],
  predicted: Sentiment[],
  positiveProbabilities?: number[]
): ModelMetrics {
  let tp = 0; // Ground truth positive, predicted positive
  let tn = 0; // Ground truth negative, predicted negative
  let fp = 0; // Ground truth negative, predicted positive
  let fn = 0; // Ground truth positive, predicted negative

  const total = groundTruth.length;

  for (let i = 0; i < total; i++) {
    const actual = groundTruth[i];
    const pred = predicted[i];

    if (actual === 'positive' && pred === 'positive') {
      tp++;
    } else if (actual === 'negative' && pred === 'negative') {
      tn++;
    } else if (actual === 'negative' && pred === 'positive') {
      fp++;
    } else if (actual === 'positive' && pred === 'negative') {
      fn++;
    }
  }

  const accuracy = total > 0 ? (tp + tn) / total : 0;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  const confusionMatrix: ConfusionMatrix = {
    truePositive: tp,
    trueNegative: tn,
    falsePositive: fp,
    falseNegative: fn,
    total
  };

  let auc: number | undefined;
  if (positiveProbabilities && positiveProbabilities.length === groundTruth.length) {
    const roc = computeRocCurve(groundTruth, positiveProbabilities);
    if (roc) {
      auc = roc.auc;
    }
  }

  return {
    modelName,
    accuracy: Number((accuracy * 100).toFixed(2)),
    precision: Number((precision * 100).toFixed(2)),
    recall: Number((recall * 100).toFixed(2)),
    f1Score: Number((f1Score * 100).toFixed(2)),
    auc,
    confusionMatrix
  };
}
