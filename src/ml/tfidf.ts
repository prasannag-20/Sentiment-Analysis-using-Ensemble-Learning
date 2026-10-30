/**
 * TF-IDF Vectorizer with Sublinear Term Frequency and Smooth Inverse Document Frequency.
 * Supports sparse vector representations for fast machine learning matrix-vector operations.
 */

export interface SparseVector {
  // Map of featureIndex -> tfidfWeight
  indices: number[];
  values: number[];
}

export interface SerializedTfidf {
  vocabulary: [string, number][];
  inverseVocabulary: string[];
  idf: number[];
  minDocFreq: number;
  maxFeatures: number;
  isFitted: boolean;
}

export class TfidfVectorizer {
  private vocabulary: Map<string, number> = new Map();
  private inverseVocabulary: string[] = [];
  private idf: Float64Array = new Float64Array(0);
  private minDocFreq: number = 2;
  private maxFeatures: number = 2500;
  private isFitted: boolean = false;

  constructor(options?: { minDocFreq?: number; maxFeatures?: number }) {
    if (options?.minDocFreq !== undefined) this.minDocFreq = options.minDocFreq;
    if (options?.maxFeatures !== undefined) this.maxFeatures = options.maxFeatures;
  }

  /**
   * Fits the vectorizer on a corpus of preprocessed token arrays.
   */
  public fit(tokenizedDocs: string[][]): void {
    const numDocs = tokenizedDocs.length;
    if (numDocs === 0) return;

    // Adjust minDocFreq if dataset is small
    const effectiveMinDocFreq = numDocs < 50 ? 1 : this.minDocFreq;

    // Step 1: Count document frequency for every unique token
    const docFrequency: Map<string, number> = new Map();

    for (const doc of tokenizedDocs) {
      const uniqueInDoc = new Set(doc);
      for (const term of uniqueInDoc) {
        docFrequency.set(term, (docFrequency.get(term) || 0) + 1);
      }
    }

    // Step 2: Filter by min document frequency and sort by frequency
    const candidateTerms: { term: string; df: number }[] = [];
    for (const [term, df] of docFrequency.entries()) {
      if (df >= effectiveMinDocFreq) {
        candidateTerms.push({ term, df });
      }
    }

    // Sort by document frequency descending, then term string ascending
    candidateTerms.sort((a, b) => b.df - a.df || a.term.localeCompare(b.term));

    // Cap to maxFeatures
    const selectedTerms = candidateTerms.slice(0, this.maxFeatures);

    // Step 3: Build vocabulary and compute smoothed IDF
    this.vocabulary.clear();
    this.inverseVocabulary = [];
    const vocabSize = selectedTerms.length;
    this.idf = new Float64Array(vocabSize);

    for (let i = 0; i < vocabSize; i++) {
      const { term, df } = selectedTerms[i];
      this.vocabulary.set(term, i);
      this.inverseVocabulary.push(term);

      // Smooth IDF: ln((1 + N) / (1 + df)) + 1
      this.idf[i] = Math.log((1 + numDocs) / (1 + df)) + 1.0;
    }

    this.isFitted = true;
  }

  /**
   * Transforms a single document's tokens into a sparse L2-normalized TF-IDF vector.
   */
  public transformDoc(tokens: string[]): SparseVector {
    if (!this.isFitted || this.vocabulary.size === 0) {
      return { indices: [], values: [] };
    }

    // Step 1: Compute raw term counts for terms that exist in vocabulary
    const termCounts: Map<number, number> = new Map();
    for (const token of tokens) {
      const index = this.vocabulary.get(token);
      if (index !== undefined) {
        termCounts.set(index, (termCounts.get(index) || 0) + 1);
      }
    }

    if (termCounts.size === 0) {
      return { indices: [], values: [] };
    }

    const indices: number[] = [];
    const rawValues: number[] = [];
    let sumSquared = 0;

    // Step 2: Compute sublinear TF-IDF: (1 + ln(tf)) * idf
    // Sort by feature index ascending for O(log K) binary search lookups
    const sortedEntries = Array.from(termCounts.entries()).sort((a, b) => a[0] - b[0]);
    for (const [index, count] of sortedEntries) {
      const tf = 1.0 + Math.log(count);
      const tfidf = tf * this.idf[index];
      indices.push(index);
      rawValues.push(tfidf);
      sumSquared += tfidf * tfidf;
    }

    // Step 3: L2 normalization
    const norm = Math.sqrt(sumSquared);
    const normalizedValues = norm > 0 ? rawValues.map(v => v / norm) : rawValues;

    return {
      indices,
      values: normalizedValues
    };
  }

  /**
   * Transforms multiple documents into sparse TF-IDF vectors.
   */
  public transform(tokenizedDocs: string[][]): SparseVector[] {
    return tokenizedDocs.map(doc => this.transformDoc(doc));
  }

  /**
   * Fits and transforms in one step.
   */
  public fitTransform(tokenizedDocs: string[][]): SparseVector[] {
    this.fit(tokenizedDocs);
    return this.transform(tokenizedDocs);
  }

  public getVocabularySize(): number {
    return this.vocabulary.size;
  }

  public getFeatureName(index: number): string | undefined {
    return this.inverseVocabulary[index];
  }

  public serialize(): SerializedTfidf {
    return {
      vocabulary: Array.from(this.vocabulary.entries()),
      inverseVocabulary: [...this.inverseVocabulary],
      idf: Array.from(this.idf),
      minDocFreq: this.minDocFreq,
      maxFeatures: this.maxFeatures,
      isFitted: this.isFitted
    };
  }

  public loadState(state: SerializedTfidf): void {
    if (!state) return;
    this.vocabulary = new Map(state.vocabulary || []);
    this.inverseVocabulary = Array.isArray(state.inverseVocabulary) ? [...state.inverseVocabulary] : [];
    this.idf = new Float64Array(state.idf || []);
    this.minDocFreq = state.minDocFreq ?? 2;
    this.maxFeatures = state.maxFeatures ?? 2500;
    this.isFitted = !!state.isFitted;
  }
}
