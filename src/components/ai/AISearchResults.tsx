/**
 * AISearchResults - Display semantic search results with relevance scores
 */
import React from 'react';
import { Sparkles, ChevronRight } from 'lucide-react';
import { cn } from '@/utils/cn';
import type { SemanticSearchResult } from '@/types/rag';
import { getCategoryColor } from '@/utils/constants';

interface AISearchResultsProps {
  results: SemanticSearchResult[];
  isLoading: boolean;
  onResultClick: (conceptId: string) => void;
  selectedIndex?: number;
  onResultHover?: (index: number) => void;
  className?: string;
}

const ScoreBadge: React.FC<{ score: number }> = ({ score }) => {
  const percentage = Math.round(score * 100);
  const color =
    percentage >= 80
      ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
      : percentage >= 60
        ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
        : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400';

  return (
    <span className={cn('px-1.5 py-0.5 rounded text-xs font-medium', color)}>
      {percentage}%
    </span>
  );
};

export const AISearchResults: React.FC<AISearchResultsProps> = ({
  results,
  isLoading,
  onResultClick,
  selectedIndex = -1,
  onResultHover,
  className,
}) => {
  if (isLoading) {
    return (
      <div className={cn('p-4 text-center', className)}>
        <div className="flex items-center justify-center gap-2 text-purple-500">
          <Sparkles className="w-4 h-4 animate-pulse" />
          <span className="text-sm">Searching semantically...</span>
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className={cn('p-6 text-center', className)}>
        <Sparkles className="w-8 h-8 text-gray-300 dark:text-gray-600 mx-auto mb-2" />
        <p className="text-sm text-gray-500 dark:text-gray-400">
          No semantic matches found
        </p>
        <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
          Try different words or phrases
        </p>
      </div>
    );
  }

  return (
    <div className={cn('divide-y divide-gray-100 dark:divide-gray-800', className)}>
      {/* Header */}
      <div className="px-3 py-2 bg-purple-50 dark:bg-purple-900/20 flex items-center gap-2">
        <Sparkles className="w-4 h-4 text-purple-500" />
        <span className="text-sm font-medium text-purple-700 dark:text-purple-300">
          AI-powered results
        </span>
        <span className="text-xs text-purple-500 dark:text-purple-400">
          ({results.length} matches)
        </span>
      </div>

      {/* Results */}
      {results.map((result, index) => {
        const categoryId = result.concept.sectionId.split('.')[0];
        const categoryColors = getCategoryColor(
          ['foundations', 'data-foundation', 'learning-problem', 'optimization', 'regularization', 'evaluation'][
            parseInt(categoryId) - 1
          ] || 'foundations'
        );

        return (
          <button
            key={result.conceptId}
            onClick={() => onResultClick(result.conceptId)}
            onMouseEnter={() => onResultHover?.(index)}
            className={cn(
              'w-full text-left px-3 py-3 transition-colors',
              'flex items-start gap-3',
              selectedIndex === index
                ? 'bg-purple-50 dark:bg-purple-900/20'
                : 'hover:bg-gray-50 dark:hover:bg-gray-800'
            )}
          >
            {/* Category indicator */}
            <div
              className={cn(
                'w-1 h-10 rounded-full flex-shrink-0',
                categoryColors.bg
              )}
            />

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-medium text-gray-900 dark:text-white truncate">
                  {result.concept.name}
                </span>
                <ScoreBadge score={result.score} />
              </div>
              {result.concept.simpleExplanation && (
                <p className="text-sm text-gray-500 dark:text-gray-400 line-clamp-2 mt-0.5">
                  {result.concept.simpleExplanation}
                </p>
              )}
            </div>

            {/* Arrow */}
            <ChevronRight className="w-4 h-4 text-gray-400 flex-shrink-0 mt-1" />
          </button>
        );
      })}
    </div>
  );
};

export default AISearchResults;
