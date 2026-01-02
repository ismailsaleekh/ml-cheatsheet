/**
 * Hook for search functionality with debouncing
 */
import { useState, useCallback, useMemo, useEffect } from 'react';
import { useAppState, useAppDispatch } from '@/context/AppContext';
import { useDebounce } from './useDebounce';
import { searchConcepts, groupResultsByCategory } from '@/utils/searchHelpers';
import type { SearchResult, GroupedSearchResults } from '@/types/search';
import { SEARCH_DEBOUNCE_MS, MAX_SEARCH_RESULTS } from '@/utils/constants';

export interface UseSearchResult {
  // Query state
  query: string;
  setQuery: (query: string) => void;
  debouncedQuery: string;

  // Results
  results: SearchResult[];
  groupedResults: GroupedSearchResults;
  totalResults: number;
  isSearching: boolean;

  // Selection state
  selectedIndex: number;
  setSelectedIndex: (index: number) => void;

  // Dropdown state
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;

  // Actions
  clearSearch: () => void;
  selectResult: (index?: number) => void;
  navigateUp: () => void;
  navigateDown: () => void;
}

/**
 * Custom hook for search with debouncing, keyboard navigation, and result selection
 */
export function useSearch(): UseSearchResult {
  const { concepts, categories } = useAppState();
  const dispatch = useAppDispatch();

  // Local state
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [isOpen, setIsOpen] = useState(false);

  // Debounce the search query
  const debouncedQuery = useDebounce(query, SEARCH_DEBOUNCE_MS);

  // Compute search results
  const results = useMemo(() => {
    if (!debouncedQuery || debouncedQuery.length < 2) {
      return [];
    }
    return searchConcepts(concepts, debouncedQuery, {
      maxResults: MAX_SEARCH_RESULTS,
    });
  }, [concepts, debouncedQuery]);

  // Group results by category
  const groupedResults = useMemo(() => {
    return groupResultsByCategory(results, concepts, categories);
  }, [results, concepts, categories]);

  // Flat list of results for keyboard navigation
  const flatResults = useMemo(() => {
    const flat: SearchResult[] = [];
    for (const categoryResults of groupedResults.values()) {
      flat.push(...categoryResults);
    }
    return flat;
  }, [groupedResults]);

  // Is currently searching (has pending debounced query)
  const isSearching = query !== debouncedQuery && query.length >= 2;

  // Update app state when results change
  useEffect(() => {
    dispatch({ type: 'SET_SEARCH', payload: debouncedQuery });
    dispatch({
      type: 'SET_SEARCH_RESULTS',
      payload: results.map((r) => r.concept),
    });
  }, [dispatch, debouncedQuery, results]);

  // Reset selection when results change
  useEffect(() => {
    setSelectedIndex(results.length > 0 ? 0 : -1);
  }, [results]);

  // Open dropdown when there's a query
  useEffect(() => {
    if (query.length >= 2) {
      setIsOpen(true);
    }
  }, [query]);

  // Clear search
  const clearSearch = useCallback(() => {
    setQuery('');
    setSelectedIndex(-1);
    setIsOpen(false);
    dispatch({ type: 'CLEAR_SEARCH' });
  }, [dispatch]);

  // Select a result and open its modal
  const selectResult = useCallback(
    (index?: number) => {
      const targetIndex = index ?? selectedIndex;
      const result = flatResults[targetIndex];

      if (result) {
        // Expand ancestors and select concept
        dispatch({ type: 'EXPAND_TO', payload: result.concept.id });
        dispatch({ type: 'SELECT_CONCEPT', payload: result.concept.id });
        clearSearch();
      }
    },
    [selectedIndex, flatResults, dispatch, clearSearch]
  );

  // Navigate up in results
  const navigateUp = useCallback(() => {
    setSelectedIndex((prev) => {
      if (prev <= 0) return flatResults.length - 1;
      return prev - 1;
    });
  }, [flatResults.length]);

  // Navigate down in results
  const navigateDown = useCallback(() => {
    setSelectedIndex((prev) => {
      if (prev >= flatResults.length - 1) return 0;
      return prev + 1;
    });
  }, [flatResults.length]);

  return {
    query,
    setQuery,
    debouncedQuery,
    results,
    groupedResults,
    totalResults: results.length,
    isSearching,
    selectedIndex,
    setSelectedIndex,
    isOpen,
    setIsOpen,
    clearSearch,
    selectResult,
    navigateUp,
    navigateDown,
  };
}

export default useSearch;
