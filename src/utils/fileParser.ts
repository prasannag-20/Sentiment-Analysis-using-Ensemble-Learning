import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { ReviewItem, Sentiment } from '../types';

export interface ParseResult {
  success: boolean;
  reviews: ReviewItem[];
  missingCount: number;
  duplicateCount: number;
  totalRawRows?: number;
  error?: string;
}

export interface UnlabeledParseResult {
  success: boolean;
  reviews: { id: string; review: string }[];
  error?: string;
}

/**
 * Normalizes column header names to lowercase trimmed strings.
 */
function normalizeHeader(header: string): string {
  return header.trim().toLowerCase().replace(/[^a-z0-9]/g, '');
}

/**
 * Parses a labeled dataset file (CSV or XLSX) containing 'review' and 'sentiment'.
 */
export async function parseDatasetFile(file: File): Promise<ParseResult> {
  const fileExt = file.name.split('.').pop()?.toLowerCase();

  try {
    let rows: Record<string, any>[] = [];

    if (fileExt === 'csv' || fileExt === 'txt') {
      const text = await file.text();
      const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
      if (parsed.errors && parsed.errors.length > 0 && parsed.data.length === 0) {
        return {
          success: false,
          reviews: [],
          missingCount: 0,
          duplicateCount: 0,
          error: `Failed to parse CSV: ${parsed.errors[0]?.message || 'Unknown syntax error'}`
        };
      }
      rows = parsed.data as Record<string, any>[];
    } else if (fileExt === 'xlsx' || fileExt === 'xls') {
      const buffer = await file.arrayBuffer();
      const workbook = XLSX.read(buffer, { type: 'array' });
      const firstSheetName = workbook.SheetNames[0];
      if (!firstSheetName) {
        return { success: false, reviews: [], missingCount: 0, duplicateCount: 0, error: 'Empty Excel workbook.' };
      }
      const sheet = workbook.Sheets[firstSheetName];
      rows = XLSX.utils.sheet_to_json(sheet);
    } else {
      return {
        success: false,
        reviews: [],
        missingCount: 0,
        duplicateCount: 0,
        error: 'Unsupported file format. Please upload a CSV (.csv) or Excel (.xlsx, .xls) file.'
      };
    }

    if (rows.length === 0) {
      return { success: false, reviews: [], missingCount: 0, duplicateCount: 0, error: 'The uploaded file contains no data rows.' };
    }

    // Identify review column and sentiment column
    const firstRow = rows[0];
    const headerKeys = Object.keys(firstRow);

    let reviewColKey: string | undefined;
    let sentimentColKey: string | undefined;

    for (const key of headerKeys) {
      const norm = normalizeHeader(key);
      if (norm === 'review' || norm === 'text' || norm === 'reviews' || norm === 'comment') {
        reviewColKey = key;
      }
      if (norm === 'sentiment' || norm === 'label' || norm === 'class' || norm === 'target') {
        sentimentColKey = key;
      }
    }

    if (!reviewColKey || !sentimentColKey) {
      return {
        success: false,
        reviews: [],
        missingCount: 0,
        duplicateCount: 0,
        error: `Required columns missing. Expected 'review' and 'sentiment'. Found: [${headerKeys.join(', ')}]`
      };
    }

    const reviews: ReviewItem[] = [];
    const seenTexts = new Set<string>();
    let missingCount = 0;
    let duplicateCount = 0;

    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      const rawReview = row[reviewColKey];
      const rawSentiment = row[sentimentColKey];

      if (!rawReview || typeof rawReview !== 'string' || !rawReview.trim() || rawSentiment === undefined || rawSentiment === null) {
        missingCount++;
        continue;
      }

      const reviewText = rawReview.trim();
      const sentimentStr = String(rawSentiment).trim().toLowerCase();

      let sentiment: Sentiment;
      if (sentimentStr === 'positive' || sentimentStr === 'pos' || sentimentStr === '1' || sentimentStr === 'true') {
        sentiment = 'positive';
      } else if (sentimentStr === 'negative' || sentimentStr === 'neg' || sentimentStr === '0' || sentimentStr === 'false') {
        sentiment = 'negative';
      } else {
        missingCount++;
        continue;
      }

      const normTextKey = reviewText.toLowerCase();
      if (seenTexts.has(normTextKey)) {
        duplicateCount++;
        // We still keep or deduplicate according to clean dataset practices
      }
      seenTexts.add(normTextKey);

      reviews.push({
        id: `uploaded-${i + 1}`,
        review: reviewText,
        sentiment
      });
    }

    if (reviews.length === 0) {
      return {
        success: false,
        reviews: [],
        missingCount,
        duplicateCount,
        error: 'No valid labeled reviews could be extracted from the file.'
      };
    }

    return {
      success: true,
      reviews,
      missingCount,
      duplicateCount,
      totalRawRows: rows.length
    };
  } catch (err: any) {
    return {
      success: false,
      reviews: [],
      missingCount: 0,
      duplicateCount: 0,
      error: `Error reading file: ${err.message || 'Unknown error'}`
    };
  }
}

/**
 * Parses an unlabeled dataset file containing 'review' column for batch prediction.
 */
export async function parseUnlabeledFile(file: File): Promise<UnlabeledParseResult> {
  const fileExt = file.name.split('.').pop()?.toLowerCase();

  try {
    let rows: Record<string, any>[] = [];

    if (fileExt === 'csv' || fileExt === 'txt') {
      const text = await file.text();
      const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
      rows = parsed.data as Record<string, any>[];
    } else if (fileExt === 'xlsx' || fileExt === 'xls') {
      const buffer = await file.arrayBuffer();
      const workbook = XLSX.read(buffer, { type: 'array' });
      const firstSheetName = workbook.SheetNames[0];
      if (!firstSheetName) return { success: false, reviews: [], error: 'Empty Excel file.' };
      rows = XLSX.utils.sheet_to_json(workbook.Sheets[firstSheetName]);
    } else {
      return { success: false, reviews: [], error: 'Please upload a CSV (.csv) or Excel (.xlsx) file.' };
    }

    if (rows.length === 0) {
      return { success: false, reviews: [], error: 'File contains no rows.' };
    }

    const firstRow = rows[0];
    const headerKeys = Object.keys(firstRow);
    let reviewColKey: string | undefined;

    for (const key of headerKeys) {
      const norm = normalizeHeader(key);
      if (norm === 'review' || norm === 'text' || norm === 'reviews' || norm === 'comment') {
        reviewColKey = key;
        break;
      }
    }

    // If no explicit 'review' column found, use the first column if it contains strings
    if (!reviewColKey && headerKeys.length > 0) {
      reviewColKey = headerKeys[0];
    }

    if (!reviewColKey) {
      return { success: false, reviews: [], error: "No review column found. Expected column 'review'." };
    }

    const reviews: { id: string; review: string }[] = [];
    for (let i = 0; i < rows.length; i++) {
      const text = rows[i][reviewColKey];
      if (text && typeof text === 'string' && text.trim().length > 0) {
        reviews.push({ id: `batch-${i + 1}`, review: text.trim() });
      }
    }

    if (reviews.length === 0) {
      return { success: false, reviews: [], error: 'No valid text reviews found in file.' };
    }

    return { success: true, reviews };
  } catch (err: any) {
    return { success: false, reviews: [], error: `File read error: ${err.message}` };
  }
}

/**
 * Combines incoming reviews with existing accumulated training data,
 * removing duplicate reviews based on normalized text content.
 */
export function combineAndDeduplicateReviews(
  existingReviews: ReviewItem[],
  incomingReviews: ReviewItem[]
): {
  combined: ReviewItem[];
  newAddedCount: number;
  duplicatesRemovedCount: number;
} {
  const seenTexts = new Set<string>();

  // Register all existing review texts
  for (const item of existingReviews) {
    seenTexts.add(item.review.trim().toLowerCase().replace(/\s+/g, ' '));
  }

  const newValidItems: ReviewItem[] = [];
  let duplicatesRemovedCount = 0;

  for (let i = 0; i < incomingReviews.length; i++) {
    const item = incomingReviews[i];
    const norm = item.review.trim().toLowerCase().replace(/\s+/g, ' ');

    if (seenTexts.has(norm)) {
      duplicatesRemovedCount++;
    } else {
      seenTexts.add(norm);
      newValidItems.push({
        id: `cumul-${Date.now()}-${i}`,
        review: item.review.trim(),
        sentiment: item.sentiment
      });
    }
  }

  return {
    combined: [...existingReviews, ...newValidItems],
    newAddedCount: newValidItems.length,
    duplicatesRemovedCount
  };
}

/**
 * Exports data to CSV string and triggers browser download.
 */
export function downloadCSV(filename: string, data: Record<string, any>[]): void {
  const csv = Papa.unparse(data);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
