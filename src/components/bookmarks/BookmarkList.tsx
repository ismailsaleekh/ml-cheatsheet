/**
 * BookmarkList - List of bookmarked concepts
 */
import React from 'react';
import { Bookmark, ChevronRight } from 'lucide-react';
import { cn } from '@/utils/cn';
import { useProgressContext } from '@/context/ProgressContext';
import { useAppState, useAppDispatch } from '@/context/AppContext';
import type { Concept } from '@/types/concept';

interface BookmarkListProps {
  onConceptClick?: (conceptId: string) => void;
  maxItems?: number;
  compact?: boolean;
  className?: string;
}

export const BookmarkList: React.FC<BookmarkListProps> = ({
  onConceptClick,
  maxItems,
  compact = false,
  className,
}) => {
  const { bookmarkedConcepts } = useProgressContext();
  const { concepts } = useAppState();
  const dispatch = useAppDispatch();

  // Get bookmarked concept objects
  const bookmarkedItems: Concept[] = concepts.filter((c) =>
    bookmarkedConcepts.has(c.id)
  );

  const displayItems = maxItems
    ? bookmarkedItems.slice(0, maxItems)
    : bookmarkedItems;

  const handleClick = (conceptId: string) => {
    if (onConceptClick) {
      onConceptClick(conceptId);
    } else {
      dispatch({ type: 'EXPAND_TO', payload: conceptId });
      dispatch({ type: 'SELECT_CONCEPT', payload: conceptId });
    }
  };

  if (bookmarkedItems.length === 0) {
    return (
      <div
        className={cn(
          'flex flex-col items-center justify-center py-8 text-center',
          className
        )}
      >
        <Bookmark className="w-8 h-8 text-gray-300 dark:text-gray-600 mb-2" />
        <p className="text-sm text-gray-500 dark:text-gray-400">
          No bookmarks yet
        </p>
        <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
          Click the bookmark icon on any concept to save it here
        </p>
      </div>
    );
  }

  return (
    <div className={cn('space-y-1', className)}>
      {displayItems.map((concept) => (
        <button
          key={concept.id}
          onClick={() => handleClick(concept.id)}
          className={cn(
            'w-full flex items-center gap-3 px-3 py-2 rounded-lg',
            'text-left transition-colors',
            'hover:bg-gray-100 dark:hover:bg-gray-800',
            'focus:outline-none focus:ring-2 focus:ring-blue-500'
          )}
        >
          <Bookmark className="w-4 h-4 text-yellow-500 fill-current flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-gray-900 dark:text-white truncate">
              {concept.name}
            </div>
            {!compact && concept.simpleExplanation && (
              <p className="text-xs text-gray-500 dark:text-gray-400 truncate mt-0.5">
                {concept.simpleExplanation}
              </p>
            )}
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
        </button>
      ))}
      {maxItems && bookmarkedItems.length > maxItems && (
        <div className="px-3 py-2 text-xs text-gray-500 dark:text-gray-400">
          +{bookmarkedItems.length - maxItems} more
        </div>
      )}
    </div>
  );
};

export default BookmarkList;
