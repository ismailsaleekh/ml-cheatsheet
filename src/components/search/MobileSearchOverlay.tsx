/**
 * MobileSearchOverlay - Full-screen search for mobile devices
 */
import { useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, Search, X, Loader2 } from 'lucide-react';
import { cn } from '@/utils/cn';
import { useSearch } from '@/hooks/useSearch';
import { useFocusTrap } from '@/hooks/useFocusTrap';

interface MobileSearchOverlayProps {
  isOpen: boolean;
  onClose: () => void;
}

export const MobileSearchOverlay: React.FC<MobileSearchOverlayProps> = ({
  isOpen,
  onClose,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  const {
    query,
    setQuery,
    results,
    groupedResults,
    isSearching,
    selectedIndex,
    clearSearch,
    selectResult,
    navigateUp,
    navigateDown,
  } = useSearch();

  // Focus trap
  useFocusTrap(overlayRef, isOpen);

  // Auto-focus input when opened
  useEffect(() => {
    if (isOpen) {
      // Small delay to allow animation to start
      const timer = setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  // Lock body scroll when open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  // Handle close
  const handleClose = useCallback(() => {
    clearSearch();
    onClose();
  }, [clearSearch, onClose]);

  // Handle result selection
  const handleSelect = useCallback(
    (index: number) => {
      selectResult(index);
      handleClose();
    },
    [selectResult, handleClose]
  );

  // Handle keyboard
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
            handleSelect(selectedIndex);
          }
          break;
        case 'Escape':
          e.preventDefault();
          handleClose();
          break;
      }
    },
    [navigateDown, navigateUp, selectedIndex, handleSelect, handleClose]
  );

  const showResults = query.length >= 2;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          ref={overlayRef}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-50 bg-white dark:bg-gray-900 flex flex-col"
        >
          {/* Header */}
          <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-200 dark:border-gray-700">
            {/* Back button */}
            <button
              onClick={handleClose}
              className={cn(
                'p-2 -ml-2 rounded-lg',
                'text-gray-600 dark:text-gray-400',
                'hover:bg-gray-100 dark:hover:bg-gray-800',
                'transition-colors'
              )}
              aria-label="Close search"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>

            {/* Search input */}
            <div className="flex-1 relative">
              <div className="absolute left-3 top-1/2 -translate-y-1/2">
                {isSearching ? (
                  <Loader2 className="w-4 h-4 text-gray-400 animate-spin" />
                ) : (
                  <Search className="w-4 h-4 text-gray-400" />
                )}
              </div>
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Search concepts..."
                className={cn(
                  'w-full pl-10 pr-10 py-2.5 rounded-lg',
                  'bg-gray-100 dark:bg-gray-800',
                  'border border-transparent',
                  'focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20',
                  'text-base text-gray-900 dark:text-white',
                  'placeholder:text-gray-500 dark:placeholder:text-gray-400'
                )}
                aria-label="Search concepts"
              />
              {query && (
                <button
                  onClick={clearSearch}
                  className={cn(
                    'absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded',
                    'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300',
                    'hover:bg-gray-200 dark:hover:bg-gray-700'
                  )}
                  aria-label="Clear search"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>

          {/* Results area */}
          <div className="flex-1 overflow-y-auto">
            {showResults ? (
              <div className="relative">
                {/* Results count */}
                <div className="px-4 py-2 bg-gray-50 dark:bg-gray-800/50 sticky top-0">
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {results.length} result{results.length !== 1 ? 's' : ''}{' '}
                    for "{query}"
                  </span>
                </div>

                {/* No results */}
                {!isSearching && results.length === 0 && (
                  <div className="p-8 text-center">
                    <p className="text-gray-500 dark:text-gray-400">
                      No results found for "{query}"
                    </p>
                    <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">
                      Try a different search term
                    </p>
                  </div>
                )}

                {/* Results list */}
                {results.length > 0 && (
                  <div className="divide-y divide-gray-100 dark:divide-gray-800">
                    {(() => {
                      let globalIndex = 0;
                      const orderedGroups: Array<{
                        category: { id: string; name: string; order: number };
                        results: typeof results;
                      }> = [];

                      for (const [category, categoryResults] of groupedResults) {
                        orderedGroups.push({ category, results: categoryResults });
                      }

                      orderedGroups.sort((a, b) => a.category.order - b.category.order);

                      return orderedGroups.map(({ category, results: categoryResults }) => (
                        <div key={category.id} className="py-2">
                          <div className="px-4 py-1.5 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                            {category.name}
                          </div>
                          <div className="px-2">
                            {categoryResults.map((result) => {
                              const currentIndex = globalIndex++;
                              const isSelected = currentIndex === selectedIndex;
                              return (
                                <button
                                  key={result.concept.id}
                                  onClick={() => handleSelect(currentIndex)}
                                  className={cn(
                                    'w-full text-left px-3 py-3 rounded-lg',
                                    'transition-colors',
                                    isSelected
                                      ? 'bg-blue-50 dark:bg-blue-900/30'
                                      : ''
                                  )}
                                >
                                  <div className="font-medium text-gray-900 dark:text-white">
                                    {result.concept.name}
                                  </div>
                                  {result.concept.simpleExplanation && (
                                    <p className="text-sm text-gray-500 dark:text-gray-400 line-clamp-2 mt-0.5">
                                      {result.concept.simpleExplanation}
                                    </p>
                                  )}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      ));
                    })()}
                  </div>
                )}
              </div>
            ) : (
              // Empty state / hints
              <div className="p-8 text-center">
                <Search className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
                <p className="text-gray-500 dark:text-gray-400">
                  Search for ML concepts
                </p>
                <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">
                  Type at least 2 characters to search
                </p>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default MobileSearchOverlay;
