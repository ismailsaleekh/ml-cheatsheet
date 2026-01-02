/**
 * SearchInput - Main search input component with dropdown results
 */
import { useRef, useCallback, useEffect } from 'react';
import { AnimatePresence } from 'framer-motion';
import { Search, X, Loader2 } from 'lucide-react';
import { cn } from '@/utils/cn';
import { SearchResults } from './SearchResults';
import { useSearch } from '@/hooks/useSearch';

interface SearchInputProps {
  placeholder?: string;
  className?: string;
  onFocus?: () => void;
  onBlur?: () => void;
}

export const SearchInput: React.FC<SearchInputProps> = ({
  placeholder = 'Search concepts...',
  className,
  onFocus,
  onBlur,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const {
    query,
    setQuery,
    results,
    groupedResults,
    isSearching,
    selectedIndex,
    setSelectedIndex,
    isOpen,
    setIsOpen,
    clearSearch,
    selectResult,
    navigateUp,
    navigateDown,
  } = useSearch();

  // Handle input change
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setQuery(e.target.value);
    },
    [setQuery]
  );

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          navigateDown();
          break;
        case 'ArrowUp':
          e.preventDefault();
          navigateUp();
          break;
        case 'Enter':
          e.preventDefault();
          if (selectedIndex >= 0) {
            selectResult(selectedIndex);
          }
          break;
        case 'Escape':
          e.preventDefault();
          if (query) {
            clearSearch();
          } else {
            inputRef.current?.blur();
          }
          break;
      }
    },
    [navigateDown, navigateUp, selectedIndex, selectResult, query, clearSearch]
  );

  // Handle focus
  const handleFocus = useCallback(() => {
    if (query.length >= 2) {
      setIsOpen(true);
    }
    onFocus?.();
  }, [query, setIsOpen, onFocus]);

  // Handle blur
  const handleBlur = useCallback(() => {
    onBlur?.();
  }, [onBlur]);

  // Click outside to close
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [setIsOpen]);

  // Focus input on / or Ctrl+K
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if already in an input
      const target = e.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return;
      }

      // Focus on / key
      if (e.key === '/') {
        e.preventDefault();
        inputRef.current?.focus();
      }

      // Focus on Ctrl+K or Cmd+K
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, []);

  const showResults = isOpen && query.length >= 2;

  return (
    <div ref={containerRef} className={cn('relative', className)}>
      {/* Input container */}
      <div className="relative">
        {/* Search icon or loading spinner */}
        <div className="absolute left-3 top-1/2 -translate-y-1/2">
          {isSearching ? (
            <Loader2 className="w-4 h-4 text-gray-400 animate-spin" />
          ) : (
            <Search className="w-4 h-4 text-gray-400" />
          )}
        </div>

        {/* Input field */}
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholder={placeholder}
          className={cn(
            'w-full pl-10 pr-20 py-2 rounded-lg',
            'bg-gray-100 dark:bg-gray-800',
            'border border-transparent',
            'focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20',
            'text-sm text-gray-900 dark:text-white',
            'placeholder:text-gray-500 dark:placeholder:text-gray-400',
            'transition-colors duration-200'
          )}
          aria-label="Search concepts"
          aria-expanded={showResults}
          aria-controls="search-results"
          aria-autocomplete="list"
          role="combobox"
        />

        {/* Right side: clear button and shortcut hint */}
        <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
          {query ? (
            <button
              onClick={clearSearch}
              className={cn(
                'p-1 rounded-md',
                'text-gray-400 hover:text-gray-600',
                'dark:hover:text-gray-300',
                'hover:bg-gray-200 dark:hover:bg-gray-700',
                'transition-colors'
              )}
              aria-label="Clear search"
            >
              <X className="w-4 h-4" />
            </button>
          ) : (
            <kbd
              className={cn(
                'hidden sm:inline-flex items-center gap-0.5',
                'px-1.5 py-0.5 rounded',
                'text-[10px] font-medium',
                'bg-gray-200 dark:bg-gray-700',
                'text-gray-500 dark:text-gray-400'
              )}
            >
              <span className="text-xs">⌘</span>K
            </kbd>
          )}
        </div>
      </div>

      {/* Search results dropdown */}
      <AnimatePresence>
        {showResults && (
          <SearchResults
            results={results}
            groupedResults={groupedResults}
            query={query}
            selectedIndex={selectedIndex}
            onSelect={selectResult}
            onHover={setSelectedIndex}
            isLoading={isSearching}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default SearchInput;
