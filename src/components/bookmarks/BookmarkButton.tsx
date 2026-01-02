/**
 * BookmarkButton - Toggle bookmark button for concepts
 */
import React from 'react';
import { Bookmark } from 'lucide-react';
import { cn } from '@/utils/cn';

interface BookmarkButtonProps {
  isBookmarked: boolean;
  onToggle: () => void;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  className?: string;
}

const SIZE_CLASSES = {
  sm: 'w-4 h-4',
  md: 'w-5 h-5',
  lg: 'w-6 h-6',
};

const BUTTON_SIZE_CLASSES = {
  sm: 'p-1.5',
  md: 'p-2',
  lg: 'p-2.5',
};

export const BookmarkButton: React.FC<BookmarkButtonProps> = ({
  isBookmarked,
  onToggle,
  size = 'md',
  showLabel = false,
  className,
}) => {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onToggle();
      }}
      className={cn(
        'flex items-center gap-2 rounded-lg transition-all duration-200',
        BUTTON_SIZE_CLASSES[size],
        isBookmarked
          ? 'text-yellow-500 hover:text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20 hover:bg-yellow-100 dark:hover:bg-yellow-900/30'
          : 'text-gray-400 hover:text-yellow-500 hover:bg-gray-100 dark:hover:bg-gray-800',
        className
      )}
      title={isBookmarked ? 'Remove bookmark' : 'Add bookmark'}
      aria-label={isBookmarked ? 'Remove bookmark' : 'Add bookmark'}
      aria-pressed={isBookmarked}
    >
      <Bookmark
        className={cn(SIZE_CLASSES[size], isBookmarked && 'fill-current')}
      />
      {showLabel && (
        <span className="text-sm">
          {isBookmarked ? 'Bookmarked' : 'Bookmark'}
        </span>
      )}
    </button>
  );
};

export default BookmarkButton;
