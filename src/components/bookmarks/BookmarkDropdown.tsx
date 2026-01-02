/**
 * BookmarkDropdown - Dropdown menu showing bookmarked concepts
 */
import React, { useRef, useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bookmark } from 'lucide-react';
import { cn } from '@/utils/cn';
import { useProgressContext } from '@/context/ProgressContext';
import { BookmarkList } from './BookmarkList';

interface BookmarkDropdownProps {
  className?: string;
}

export const BookmarkDropdown: React.FC<BookmarkDropdownProps> = ({
  className,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const { bookmarkedConcepts } = useProgressContext();

  const bookmarkCount = bookmarkedConcepts.size;

  // Close on click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  // Close on escape
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
    }
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen]);

  return (
    <div ref={dropdownRef} className={cn('relative', className)}>
      {/* Toggle button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'relative p-2 rounded-lg transition-colors duration-200',
          'text-gray-600 dark:text-gray-400',
          'hover:bg-gray-100 dark:hover:bg-gray-800',
          'focus:outline-none focus:ring-2 focus:ring-blue-500',
          isOpen && 'bg-gray-100 dark:bg-gray-800'
        )}
        title="Bookmarks"
        aria-label={`Bookmarks (${bookmarkCount})`}
        aria-expanded={isOpen}
        aria-haspopup="true"
      >
        <Bookmark className="w-5 h-5" />
        {bookmarkCount > 0 && (
          <span
            className={cn(
              'absolute -top-1 -right-1',
              'min-w-[18px] h-[18px] px-1',
              'flex items-center justify-center',
              'text-xs font-medium text-white',
              'bg-yellow-500 rounded-full'
            )}
          >
            {bookmarkCount > 99 ? '99+' : bookmarkCount}
          </span>
        )}
      </button>

      {/* Dropdown */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className={cn(
              'absolute right-0 top-full mt-2 z-50',
              'w-80 max-h-96 overflow-y-auto',
              'bg-white dark:bg-gray-900',
              'rounded-xl shadow-xl',
              'border border-gray-200 dark:border-gray-700'
            )}
          >
            {/* Header */}
            <div className="sticky top-0 px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                Bookmarks
              </h3>
              {bookmarkCount > 0 && (
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                  {bookmarkCount} saved concept{bookmarkCount !== 1 ? 's' : ''}
                </p>
              )}
            </div>

            {/* List */}
            <div className="p-2">
              <BookmarkList
                onConceptClick={() => setIsOpen(false)}
                maxItems={10}
                compact
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default BookmarkDropdown;
