/**
 * ModalNavigation - Previous/Next navigation buttons
 */
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { cn } from '@/utils/cn';

interface ModalNavigationProps {
  previousConcept: { id: string; name: string } | null;
  nextConcept: { id: string; name: string } | null;
  onPrevious: () => void;
  onNext: () => void;
}

export const ModalNavigation: React.FC<ModalNavigationProps> = ({
  previousConcept,
  nextConcept,
  onPrevious,
  onNext,
}) => {
  return (
    <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
      {/* Previous button */}
      <button
        onClick={onPrevious}
        disabled={!previousConcept}
        className={cn(
          'flex items-center gap-2 px-4 py-2 rounded-lg',
          'text-sm font-medium transition-colors duration-150',
          previousConcept
            ? 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
            : 'text-gray-400 dark:text-gray-600 cursor-not-allowed',
          'focus:outline-none focus:ring-2 focus:ring-blue-500'
        )}
        aria-label={previousConcept ? `Go to ${previousConcept.name}` : 'No previous concept'}
      >
        <ChevronLeft className="w-5 h-5" />
        <div className="text-left hidden sm:block">
          <span className="block text-xs text-gray-500 dark:text-gray-400">Previous</span>
          <span className="block truncate max-w-[150px]">
            {previousConcept?.name || 'None'}
          </span>
        </div>
        <span className="sm:hidden">Previous</span>
      </button>

      {/* Keyboard hint */}
      <div className="hidden md:flex items-center gap-1 text-xs text-gray-400 dark:text-gray-500">
        <span>Use</span>
        <kbd className="px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 font-mono">
          ←
        </kbd>
        <kbd className="px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 font-mono">
          →
        </kbd>
        <span>to navigate</span>
      </div>

      {/* Next button */}
      <button
        onClick={onNext}
        disabled={!nextConcept}
        className={cn(
          'flex items-center gap-2 px-4 py-2 rounded-lg',
          'text-sm font-medium transition-colors duration-150',
          nextConcept
            ? 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
            : 'text-gray-400 dark:text-gray-600 cursor-not-allowed',
          'focus:outline-none focus:ring-2 focus:ring-blue-500'
        )}
        aria-label={nextConcept ? `Go to ${nextConcept.name}` : 'No next concept'}
      >
        <span className="sm:hidden">Next</span>
        <div className="text-right hidden sm:block">
          <span className="block text-xs text-gray-500 dark:text-gray-400">Next</span>
          <span className="block truncate max-w-[150px]">
            {nextConcept?.name || 'None'}
          </span>
        </div>
        <ChevronRight className="w-5 h-5" />
      </button>
    </div>
  );
};

export default ModalNavigation;
