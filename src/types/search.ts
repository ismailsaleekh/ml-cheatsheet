/**
 * Search-related type definitions
 */
import type { Concept, Category } from './concept';

/**
 * Match range for highlighting search results
 */
export interface MatchRange {
  start: number;
  end: number;
}

/**
 * Field that matched the search query
 */
export type MatchedField = 'name' | 'tags' | 'content';

/**
 * Search result with scoring and match information
 */
export interface SearchResult {
  concept: Concept;
  score: number;
  matchedField: MatchedField;
  matchRanges: MatchRange[];
}

/**
 * Options for search behavior
 */
export interface SearchOptions {
  maxResults?: number;
  minQueryLength?: number;
  searchName?: boolean;
  searchTags?: boolean;
  searchContent?: boolean;
}

/**
 * Default search options
 */
export const DEFAULT_SEARCH_OPTIONS: Required<SearchOptions> = {
  maxResults: 20,
  minQueryLength: 2,
  searchName: true,
  searchTags: true,
  searchContent: true,
};

/**
 * Grouped search results by category
 */
export type GroupedSearchResults = Map<Category, SearchResult[]>;

/**
 * Search state for the useSearch hook
 */
export interface SearchState {
  query: string;
  results: SearchResult[];
  isSearching: boolean;
  selectedIndex: number;
  isOpen: boolean;
}
