/**
 * SearchResultItem - Individual search result item
 */
import { useRef, useEffect } from 'react';
import { cn } from '@/utils/cn';
import { SearchHighlight } from './SearchHighlight';
import { Badge } from '@/components/common/Badge';
import type { SearchResult } from '@/types/search';
import { getCategoryColor } from '@/utils/constants';

interface SearchResultItemProps {
  result: SearchResult;
  query: string;
  isSelected: boolean;
  categoryId?: string;
  onClick: () => void;
  onMouseEnter: () => void;
}

export const SearchResultItem: React.FC<SearchResultItemProps> = ({
  result,
  query,
  isSelected,
  categoryId,
  onClick,
  onMouseEnter,
}) => {
  const itemRef = useRef<HTMLButtonElement>(null);
  const { concept, matchedField } = result;

  // Scroll selected item into view
  useEffect(() => {
    if (isSelected && itemRef.current) {
      itemRef.current.scrollIntoView({
        block: 'nearest',
        behavior: 'smooth',
      });
    }
  }, [isSelected]);

  // Get category color
  const categoryColors = categoryId ? getCategoryColor(categoryId) : null;

  // Get a brief description (truncated)
  const description = concept.simpleExplanation
    ? concept.simpleExplanation.slice(0, 80) + (concept.simpleExplanation.length > 80 ? '...' : '')
    : '';

  return (
    <button
      ref={itemRef}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      className={cn(
        'w-full text-left px-3 py-2 rounded-lg',
        'transition-colors duration-100',
        'focus:outline-none',
        isSelected
          ? 'bg-blue-50 dark:bg-blue-900/30'
          : 'hover:bg-gray-50 dark:hover:bg-gray-800'
      )}
    >
      <div className="flex items-start gap-3">
        {/* Category color indicator */}
        {categoryColors && (
          <div
            className={cn(
              'w-1 h-8 rounded-full flex-shrink-0 mt-0.5',
              categoryColors.bg
            )}
          />
        )}

        <div className="flex-1 min-w-0">
          {/* Concept name with highlighting */}
          <div className="flex items-center gap-2">
            <SearchHighlight
              text={concept.name}
              query={query}
              className={cn(
                'font-medium text-gray-900 dark:text-white',
                'truncate'
              )}
            />
            {matchedField === 'tags' && (
              <span className="text-xs text-gray-400">(tag match)</span>
            )}
            {matchedField === 'content' && (
              <span className="text-xs text-gray-400">(content match)</span>
            )}
          </div>

          {/* Brief description */}
          {description && (
            <p className="text-sm text-gray-500 dark:text-gray-400 truncate mt-0.5">
              {description}
            </p>
          )}

          {/* Badges */}
          <div className="flex items-center gap-2 mt-1">
            <Badge variant="section" className="text-[10px] px-1.5 py-0">
              {concept.sectionId}
            </Badge>
            <Badge variant="difficulty" difficulty={concept.difficulty} className="text-[10px] px-1.5 py-0">
              {concept.difficulty}
            </Badge>
          </div>
        </div>
      </div>
    </button>
  );
};

export default SearchResultItem;
