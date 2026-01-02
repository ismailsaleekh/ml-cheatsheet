/**
 * Hook for semantic search functionality
 */
import { useState, useCallback, useEffect } from 'react';
import { useDebounce } from './useDebounce';
import { ragService } from '@/services/ragService';
import type { SemanticSearchResult, SearchMode } from '@/types/rag';

const SEMANTIC_SEARCH_DEBOUNCE_MS = 300;

export interface UseSemanticSearchResult {
  // Query state
  query: string;
  setQuery: (query: string) => void;
  debouncedQuery: string;

  // Results
  results: SemanticSearchResult[];
  isSearching: boolean;
  error: string | null;

  // Search mode
  searchMode: SearchMode;
  setSearchMode: (mode: SearchMode) => void;
  isSemanticAvailable: boolean;

  // Actions
  search: (query: string) => Promise<void>;
  clearResults: () => void;
}

/**
 * Hook for semantic search with debouncing
 */
export function useSemanticSearch(): UseSemanticSearchResult {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SemanticSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchMode, setSearchMode] = useState<SearchMode>('keyword');

  const debouncedQuery = useDebounce(query, SEMANTIC_SEARCH_DEBOUNCE_MS);

  // Check if semantic search is available
  const isSemanticAvailable = ragService.isReady();

  // Perform semantic search
  const search = useCallback(async (searchQuery: string) => {
    if (!searchQuery || searchQuery.length < 2) {
      setResults([]);
      return;
    }

    if (searchMode !== 'semantic' || !isSemanticAvailable) {
      return;
    }

    setIsSearching(true);
    setError(null);

    try {
      const searchResults = await ragService.semanticSearch(searchQuery, 20);
      setResults(searchResults);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed');
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  }, [searchMode, isSemanticAvailable]);

  // Auto-search when debounced query changes
  useEffect(() => {
    if (searchMode === 'semantic' && debouncedQuery) {
      search(debouncedQuery);
    }
  }, [debouncedQuery, searchMode, search]);

  // Clear results
  const clearResults = useCallback(() => {
    setQuery('');
    setResults([]);
    setError(null);
  }, []);

  return {
    query,
    setQuery,
    debouncedQuery,
    results,
    isSearching,
    error,
    searchMode,
    setSearchMode,
    isSemanticAvailable,
    search,
    clearResults,
  };
}

export default useSemanticSearch;
