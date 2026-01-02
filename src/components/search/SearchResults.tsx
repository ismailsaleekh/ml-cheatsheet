/**
 * SearchResults - Dropdown showing categorized search results
 */
import { motion } from 'framer-motion';
import { cn } from '@/utils/cn';
import { SearchResultItem } from './SearchResultItem';
import type { SearchResult, GroupedSearchResults } from '@/types/search';
import type { Category } from '@/types/concept';

interface SearchResultsProps {
  results: SearchResult[];
  groupedResults: GroupedSearchResults;
  query: string;
  selectedIndex: number;
  onSelect: (index: number) => void;
  onHover: (index: number) => void;
  isLoading?: boolean;
}

export const SearchResults: React.FC<SearchResultsProps> = ({
  results,
  groupedResults,
  query,
  selectedIndex,
  onSelect,
  onHover,
  isLoading,
}) => {
  // Track global index for keyboard navigation
  let globalIndex = 0;

  // Convert grouped results to ordered array
  const orderedGroups: Array<{ category: Category; results: SearchResult[] }> = [];
  for (const [category, categoryResults] of groupedResults) {
    orderedGroups.push({ category, results: categoryResults });
  }

  // Sort groups by category order
  orderedGroups.sort((a, b) => a.category.order - b.category.order);

  // Empty state
  if (!isLoading && results.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className={cn(
          'absolute top-full left-0 right-0 mt-2 z-50',
          'bg-white dark:bg-gray-900',
          'rounded-xl shadow-xl border border-gray-200 dark:border-gray-700',
          'p-4 text-center'
        )}
      >
        <p className="text-gray-500 dark:text-gray-400">
          No results found for "{query}"
        </p>
        <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">
          Try a different search term
        </p>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.15 }}
      className={cn(
        'absolute top-full left-0 right-0 mt-2 z-50',
        'bg-white dark:bg-gray-900',
        'rounded-xl shadow-xl border border-gray-200 dark:border-gray-700',
        'max-h-[60vh] overflow-y-auto',
        'divide-y divide-gray-100 dark:divide-gray-800'
      )}
    >
      {/* Results count header */}
      <div className="px-4 py-2 bg-gray-50 dark:bg-gray-800/50 rounded-t-xl">
        <span className="text-sm text-gray-500 dark:text-gray-400">
          {results.length} result{results.length !== 1 ? 's' : ''} for "{query}"
        </span>
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="p-4 flex items-center justify-center">
          <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <span className="ml-2 text-sm text-gray-500">Searching...</span>
        </div>
      )}

      {/* Grouped results */}
      {!isLoading && orderedGroups.map(({ category, results: categoryResults }) => (
        <div key={category.id} className="py-2">
          {/* Category header */}
          <div className="px-4 py-1.5 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
            {category.name}
          </div>

          {/* Results in this category */}
          <div className="px-2">
            {categoryResults.map((result) => {
              const currentIndex = globalIndex++;
              return (
                <SearchResultItem
                  key={result.concept.id}
                  result={result}
                  query={query}
                  isSelected={currentIndex === selectedIndex}
                  categoryId={category.id}
                  onClick={() => onSelect(currentIndex)}
                  onMouseEnter={() => onHover(currentIndex)}
                />
              );
            })}
          </div>
        </div>
      ))}

      {/* Keyboard hints footer */}
      <div className="px-4 py-2 bg-gray-50 dark:bg-gray-800/50 rounded-b-xl">
        <div className="flex items-center justify-center gap-4 text-xs text-gray-400 dark:text-gray-500">
          <span>
            <kbd className="px-1.5 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-[10px]">↑</kbd>
            <kbd className="px-1.5 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-[10px] ml-0.5">↓</kbd>
            <span className="ml-1">Navigate</span>
          </span>
          <span>
            <kbd className="px-1.5 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-[10px]">⏎</kbd>
            <span className="ml-1">Select</span>
          </span>
          <span>
            <kbd className="px-1.5 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-[10px]">Esc</kbd>
            <span className="ml-1">Close</span>
          </span>
        </div>
      </div>
    </motion.div>
  );
};

export default SearchResults;
