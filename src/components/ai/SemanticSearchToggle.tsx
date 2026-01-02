/**
 * SemanticSearchToggle - Toggle between keyword and semantic search
 */
import React from 'react';
import { Search, Sparkles } from 'lucide-react';
import { cn } from '@/utils/cn';
import type { SearchMode } from '@/types/rag';

interface SemanticSearchToggleProps {
  mode: SearchMode;
  onModeChange: (mode: SearchMode) => void;
  isSemanticAvailable: boolean;
  className?: string;
}

export const SemanticSearchToggle: React.FC<SemanticSearchToggleProps> = ({
  mode,
  onModeChange,
  isSemanticAvailable,
  className,
}) => {
  return (
    <div className={cn('flex items-center gap-1 p-1 bg-gray-100 dark:bg-gray-800 rounded-lg', className)}>
      <button
        onClick={() => onModeChange('keyword')}
        className={cn(
          'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all',
          mode === 'keyword'
            ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
            : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
        )}
        title="Keyword search"
      >
        <Search className="w-3.5 h-3.5" />
        <span>Keyword</span>
      </button>
      <button
        onClick={() => isSemanticAvailable && onModeChange('semantic')}
        disabled={!isSemanticAvailable}
        className={cn(
          'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all',
          mode === 'semantic'
            ? 'bg-white dark:bg-gray-700 text-purple-600 dark:text-purple-400 shadow-sm'
            : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300',
          !isSemanticAvailable && 'opacity-50 cursor-not-allowed'
        )}
        title={isSemanticAvailable ? 'Semantic search (AI-powered)' : 'Semantic search unavailable'}
      >
        <Sparkles className="w-3.5 h-3.5" />
        <span>Semantic</span>
      </button>
    </div>
  );
};

export default SemanticSearchToggle;
