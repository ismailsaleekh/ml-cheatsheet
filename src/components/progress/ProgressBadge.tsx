/**
 * ProgressBadge - Shows learned/bookmark status on concept blocks
 */
import React from 'react';
import { Check, Bookmark } from 'lucide-react';
import { cn } from '@/utils/cn';

interface ProgressBadgeProps {
  isLearned: boolean;
  isBookmarked: boolean;
  size?: 'sm' | 'md';
  className?: string;
}

export const ProgressBadge: React.FC<ProgressBadgeProps> = ({
  isLearned,
  isBookmarked,
  size = 'sm',
  className,
}) => {
  if (!isLearned && !isBookmarked) {
    return null;
  }

  const iconSize = size === 'sm' ? 'w-3 h-3' : 'w-4 h-4';

  return (
    <div className={cn('flex items-center gap-1', className)}>
      {isLearned && (
        <div
          className={cn(
            'flex items-center justify-center rounded-full',
            'bg-green-100 dark:bg-green-900/30',
            size === 'sm' ? 'w-5 h-5' : 'w-6 h-6'
          )}
          title="Learned"
        >
          <Check className={cn(iconSize, 'text-green-600 dark:text-green-400')} />
        </div>
      )}
      {isBookmarked && (
        <div
          className={cn(
            'flex items-center justify-center rounded-full',
            'bg-yellow-100 dark:bg-yellow-900/30',
            size === 'sm' ? 'w-5 h-5' : 'w-6 h-6'
          )}
          title="Bookmarked"
        >
          <Bookmark
            className={cn(iconSize, 'text-yellow-600 dark:text-yellow-500 fill-current')}
          />
        </div>
      )}
    </div>
  );
};

export default ProgressBadge;
