/**
 * Search helper functions for concept searching
 */
import type { Concept, Category } from '@/types/concept';
import type {
  SearchResult,
  SearchOptions,
  MatchRange,
  GroupedSearchResults,
} from '@/types/search';
import { getCategoryForConcept } from './dataHelpers';

/**
 * Normalize text for searching (lowercase, trim whitespace)
 */
export function normalizeText(text: string): string {
  return text.toLowerCase().trim();
}

/**
 * Find all match ranges in text for a query
 */
export function findMatchRanges(text: string, query: string): MatchRange[] {
  const ranges: MatchRange[] = [];
  const normalizedText = normalizeText(text);
  const normalizedQuery = normalizeText(query);

  if (!normalizedQuery) return ranges;

  let startIndex = 0;
  let index: number;

  while ((index = normalizedText.indexOf(normalizedQuery, startIndex)) !== -1) {
    ranges.push({
      start: index,
      end: index + normalizedQuery.length,
    });
    startIndex = index + 1;
  }

  return ranges;
}

/**
 * Calculate search score for a concept
 * Higher scores = better matches
 */
export function calculateScore(
  concept: Concept,
  query: string,
  options: Required<SearchOptions>
): { score: number; matchedField: 'name' | 'tags' | 'content'; matchRanges: MatchRange[] } {
  const normalizedQuery = normalizeText(query);
  let bestScore = 0;
  let matchedField: 'name' | 'tags' | 'content' = 'name';
  let matchRanges: MatchRange[] = [];

  // Check name match (highest priority)
  if (options.searchName) {
    const normalizedName = normalizeText(concept.name);

    // Exact match
    if (normalizedName === normalizedQuery) {
      return {
        score: 100,
        matchedField: 'name',
        matchRanges: [{ start: 0, end: concept.name.length }],
      };
    }

    // Name starts with query
    if (normalizedName.startsWith(normalizedQuery)) {
      const score = 90;
      if (score > bestScore) {
        bestScore = score;
        matchedField = 'name';
        matchRanges = [{ start: 0, end: query.length }];
      }
    }

    // Name contains query
    const nameIndex = normalizedName.indexOf(normalizedQuery);
    if (nameIndex !== -1) {
      const score = 80;
      if (score > bestScore) {
        bestScore = score;
        matchedField = 'name';
        matchRanges = findMatchRanges(concept.name, query);
      }
    }

    // Word boundary match in name
    const words = normalizedName.split(/\s+/);
    for (const word of words) {
      if (word.startsWith(normalizedQuery)) {
        const score = 75;
        if (score > bestScore) {
          bestScore = score;
          matchedField = 'name';
          matchRanges = findMatchRanges(concept.name, query);
        }
        break;
      }
    }
  }

  // Check tag matches (medium priority)
  if (options.searchTags && concept.tags.length > 0) {
    for (const tag of concept.tags) {
      const normalizedTag = normalizeText(tag);

      // Exact tag match
      if (normalizedTag === normalizedQuery) {
        const score = 60;
        if (score > bestScore) {
          bestScore = score;
          matchedField = 'tags';
          matchRanges = [{ start: 0, end: tag.length }];
        }
        break;
      }

      // Tag contains query
      if (normalizedTag.includes(normalizedQuery)) {
        const score = 50;
        if (score > bestScore) {
          bestScore = score;
          matchedField = 'tags';
          matchRanges = findMatchRanges(tag, query);
        }
      }
    }
  }

  // Check content matches (lowest priority)
  if (options.searchContent) {
    const contentFields = [concept.fullExplanation, concept.simpleExplanation];

    for (const content of contentFields) {
      if (!content) continue;
      const normalizedContent = normalizeText(content);

      if (normalizedContent.includes(normalizedQuery)) {
        const score = 30;
        if (score > bestScore) {
          bestScore = score;
          matchedField = 'content';
          matchRanges = findMatchRanges(content, query);
        }
        break;
      }
    }
  }

  return { score: bestScore, matchedField, matchRanges };
}

/**
 * Search concepts with the given query
 */
export function searchConcepts(
  concepts: Concept[],
  query: string,
  options: Partial<SearchOptions> = {}
): SearchResult[] {
  const opts: Required<SearchOptions> = {
    maxResults: options.maxResults ?? 20,
    minQueryLength: options.minQueryLength ?? 2,
    searchName: options.searchName ?? true,
    searchTags: options.searchTags ?? true,
    searchContent: options.searchContent ?? true,
  };

  const normalizedQuery = normalizeText(query);

  // Return empty if query is too short
  if (normalizedQuery.length < opts.minQueryLength) {
    return [];
  }

  const results: SearchResult[] = [];

  for (const concept of concepts) {
    const { score, matchedField, matchRanges } = calculateScore(concept, query, opts);

    if (score > 0) {
      results.push({
        concept,
        score,
        matchedField,
        matchRanges,
      });
    }
  }

  // Sort by score descending
  results.sort((a, b) => b.score - a.score);

  // Limit results
  return results.slice(0, opts.maxResults);
}

/**
 * Group search results by category
 */
export function groupResultsByCategory(
  results: SearchResult[],
  concepts: Concept[],
  categories: Category[]
): GroupedSearchResults {
  const grouped: GroupedSearchResults = new Map();

  for (const result of results) {
    const category = getCategoryForConcept(concepts, categories, result.concept.id);

    if (category) {
      if (!grouped.has(category)) {
        grouped.set(category, []);
      }
      grouped.get(category)!.push(result);
    }
  }

  return grouped;
}

/**
 * Get flat list of concept IDs from grouped results
 */
export function getFlatResultIds(grouped: GroupedSearchResults): string[] {
  const ids: string[] = [];

  for (const results of grouped.values()) {
    for (const result of results) {
      ids.push(result.concept.id);
    }
  }

  return ids;
}
