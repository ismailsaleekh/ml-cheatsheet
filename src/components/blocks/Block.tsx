/**
 * Block component - Core interactive element for navigation
 */
import React from 'react';
import { motion } from 'framer-motion';
import { ChevronDown, Info, Check, Bookmark } from 'lucide-react';
import type { Concept } from '@/types/concept';
import { cn } from '@/utils/cn';
import { getCategoryColor, getDifficultyColor, DEPTH_INDENT } from '@/utils/constants';
import { useProgressContext } from '@/context/ProgressContext';

interface BlockProps {
  concept: Concept;
  categoryId: string;
  isExpanded: boolean;
  hasChildren: boolean;
  onToggle: () => void;
  onViewDetails: () => void;
  depth?: number;
}

export const Block: React.FC<BlockProps> = ({
  concept,
  categoryId,
  isExpanded,
  hasChildren,
  onToggle,
  onViewDetails,
  depth = 0,
}) => {
  const categoryColors = getCategoryColor(categoryId);
  const difficultyClasses = getDifficultyColor(concept.difficulty);
  const { isLearned, isBookmarked, toggleLearned, toggleBookmark } = useProgressContext();

  const learned = isLearned(concept.id);
  const bookmarked = isBookmarked(concept.id);

  // Handle learn toggle
  const handleLearnToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    toggleLearned(concept.id);
  };

  // Handle bookmark toggle
  const handleBookmarkToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    toggleBookmark(concept.id);
  };

  // Handle keyboard interaction
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (hasChildren) {
        onToggle();
      }
    }
  };

  // Handle info button click
  const handleInfoClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onViewDetails();
  };

  // Handle main block click
  const handleBlockClick = () => {
    if (hasChildren) {
      onToggle();
    } else {
      onViewDetails();
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.2 }}
      className={cn(
        'relative rounded-xl bg-white dark:bg-gray-800',
        'border border-gray-200 dark:border-gray-700',
        'shadow-sm hover:shadow-md transition-all duration-200',
        'cursor-pointer select-none',
        'border-l-4',
        categoryColors.border,
        isExpanded && [
          'ring-2 ring-offset-2 ring-offset-white dark:ring-offset-gray-900',
          categoryColors.ring,
        ]
      )}
      style={{
        marginLeft: depth > 0 ? `${depth * DEPTH_INDENT}px` : undefined,
      }}
      onClick={handleBlockClick}
      onKeyDown={handleKeyDown}
      role="button"
      aria-expanded={hasChildren ? isExpanded : undefined}
      aria-label={`${concept.name}. ${hasChildren ? (isExpanded ? 'Collapse' : 'Expand') + ' to see related concepts' : 'View details'}`}
      tabIndex={0}
    >
      <div className="p-4">
        <div className="flex items-start justify-between gap-3">
          {/* Main content */}
          <div className="flex-1 min-w-0">
            <h3 className="text-base font-semibold text-gray-900 dark:text-white truncate">
              {concept.name}
            </h3>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400 line-clamp-2">
              {concept.simpleExplanation}
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-0.5 flex-shrink-0">
            {/* Learn toggle */}
            <button
              onClick={handleLearnToggle}
              className={cn(
                'p-1.5 rounded-full transition-colors duration-150',
                learned
                  ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                  : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300',
                'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500'
              )}
              aria-label={learned ? `Mark ${concept.name} as not learned` : `Mark ${concept.name} as learned`}
              title={learned ? 'Learned' : 'Mark as learned'}
            >
              <Check className="w-4 h-4" />
            </button>

            {/* Bookmark toggle */}
            <button
              onClick={handleBookmarkToggle}
              className={cn(
                'p-1.5 rounded-full transition-colors duration-150',
                bookmarked
                  ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-500'
                  : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300',
                'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500'
              )}
              aria-label={bookmarked ? `Remove bookmark from ${concept.name}` : `Bookmark ${concept.name}`}
              title={bookmarked ? 'Bookmarked' : 'Bookmark'}
            >
              <Bookmark className={cn('w-4 h-4', bookmarked && 'fill-current')} />
            </button>

            {/* Info button */}
            <button
              onClick={handleInfoClick}
              className={cn(
                'p-1.5 rounded-full transition-colors duration-150',
                'hover:bg-gray-100 dark:hover:bg-gray-700',
                'focus:outline-none focus:ring-2 focus:ring-offset-2',
                categoryColors.ring
              )}
              aria-label={`View details for ${concept.name}`}
              title="View details"
            >
              <Info className="w-4 h-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" />
            </button>

            {hasChildren && (
              <motion.div
                animate={{ rotate: isExpanded ? 180 : 0 }}
                transition={{ duration: 0.2 }}
                className="p-1"
              >
                <ChevronDown className="w-5 h-5 text-gray-400" />
              </motion.div>
            )}
          </div>
        </div>

        {/* Badges */}
        <div className="mt-3 flex flex-wrap gap-2">
          <span
            className={cn(
              'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
              categoryColors.bgLight,
              categoryColors.text
            )}
          >
            {concept.sectionId}
          </span>
          <span
            className={cn(
              'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
              difficultyClasses
            )}
          >
            {concept.difficulty}
          </span>
          {hasChildren && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300">
              {isExpanded ? 'Click to collapse' : 'Click to expand'}
            </span>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default Block;
