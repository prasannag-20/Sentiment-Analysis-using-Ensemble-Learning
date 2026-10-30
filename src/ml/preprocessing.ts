/**
 * Text Preprocessing Pipeline for IMDb Review Sentiment Analysis
 * Handles:
 * - HTML stripping
 * - Lowercasing
 * - Unnecessary punctuation and symbol cleaning
 * - Whitespace normalization
 * - Stopword removal with negation sensitivity
 * - Negation binding (e.g. "not bad" -> "not_bad")
 * - Unigram and Bigram extraction
 */

// Standard English stopwords, excluding negation terms crucial for sentiment
const STOP_WORDS = new Set([
  'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
  'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
  'between', 'both', 'by', 'could', 'did', 'do', 'does', 'doing', 'down',
  'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
  'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his',
  'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me',
  'more', 'most', 'my', 'myself', 'of', 'off', 'on', 'once', 'only', 'or',
  'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
  'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their',
  'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this',
  'those', 'through', 'to', 'too', 'under', 'until', 'up', 'was',
  'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
  'why', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves'
]);

const NEGATION_WORDS = new Set([
  'not', 'no', 'never', 'neither', 'nor', 'hardly', 'scarcely', 'barely', "n't"
]);

/**
 * Cleans raw text string by removing HTML, lowercasing, and normalizing punctuation.
 */
export function cleanRawText(text: string): string {
  if (!text) return '';
  return text
    // Replace HTML break tags with space
    .replace(/<br\s*\/?>/gi, ' ')
    // Strip any other HTML tags
    .replace(/<[^>]*>/g, ' ')
    // Normalize contractions like won't, can't, isn't
    .replace(/n't\b/gi, ' not')
    .replace(/'re\b/gi, ' are')
    .replace(/'s\b/gi, '')
    .replace(/'d\b/gi, ' would')
    .replace(/'ll\b/gi, ' will')
    .replace(/'t\b/gi, ' not')
    .replace(/'ve\b/gi, ' have')
    .replace(/'m\b/gi, ' am')
    // Convert to lowercase
    .toLowerCase()
    // Replace punctuation with spaces while preserving basic alphanumeric characters
    .replace(/[^a-z0-9\s]/g, ' ')
    // Collapse consecutive whitespaces
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Tokenizes cleaned text into processed terms with negation binding and stopword filtering.
 */
export function tokenizeAndPreprocess(text: string): string[] {
  const cleaned = cleanRawText(text);
  if (!cleaned) return [];

  const rawTokens = cleaned.split(' ').filter(token => token.length > 0);
  const processedTokens: string[] = [];
  let isNegated = false;
  let negatedSteps = 0;

  for (const token of rawTokens) {
    if (NEGATION_WORDS.has(token)) {
      isNegated = true;
      negatedSteps = 0;
      processedTokens.push(token);
      continue;
    }

    if (isNegated) {
      negatedSteps++;
      // Bind negation prefix to the following 2 words
      const negatedToken = `not_${token}`;
      if (!STOP_WORDS.has(token) || token.length > 2) {
        processedTokens.push(negatedToken);
      }
      if (negatedSteps >= 2) {
        isNegated = false;
      }
    } else {
      if (!STOP_WORDS.has(token) && token.length > 1) {
        processedTokens.push(token);
      }
    }
  }

  return processedTokens;
}

/**
 * Generates unigrams and bigrams from tokens.
 */
export function extractNgrams(tokens: string[]): string[] {
  const ngrams: string[] = [...tokens];
  
  // Extract bigrams
  for (let i = 0; i < tokens.length - 1; i++) {
    const bigram = `${tokens[i]}_${tokens[i + 1]}`;
    ngrams.push(bigram);
  }

  return ngrams;
}

/**
 * Complete preprocessing pipeline from raw review string to n-gram tokens.
 */
export function preprocessReview(text: string): string[] {
  const tokens = tokenizeAndPreprocess(text);
  return extractNgrams(tokens);
}
