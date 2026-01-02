/**
 * ProgressPanel - Slide-out panel showing detailed progress
 */
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Download, RotateCcw } from 'lucide-react';
import { cn } from '@/utils/cn';
import { useProgressContext } from '@/context/ProgressContext';
import { useAppState } from '@/context/AppContext';
import { ProgressBar } from './ProgressBar';
import { ProgressStats } from './ProgressStats';

interface ProgressPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ProgressPanel: React.FC<ProgressPanelProps> = ({ isOpen, onClose }) => {
  const { stats, getOverallProgress, getCategoryProgress, exportProgress, resetProgress } =
    useProgressContext();
  const { concepts, categories } = useAppState();

  const overall = getOverallProgress();

  const handleExport = () => {
    const data = exportProgress();
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ml-cheatsheet-progress-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleReset = () => {
    if (window.confirm('Are you sure you want to reset all progress? This cannot be undone.')) {
      resetProgress();
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm"
            onClick={onClose}
          />

          {/* Panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className={cn(
              'fixed right-0 top-0 bottom-0 z-50',
              'w-full max-w-md',
              'bg-white dark:bg-gray-900',
              'border-l border-gray-200 dark:border-gray-700',
              'shadow-2xl',
              'flex flex-col'
            )}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Learning Progress
              </h2>
              <button
                onClick={onClose}
                className={cn(
                  'p-2 rounded-lg',
                  'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200',
                  'hover:bg-gray-100 dark:hover:bg-gray-800',
                  'transition-colors'
                )}
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
              {/* Overall Progress */}
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Overall Progress
                </h3>
                <ProgressBar
                  value={overall.learned}
                  max={overall.total}
                  size="lg"
                  showLabel
                  color="green"
                />
              </div>

              {/* Stats */}
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Statistics
                </h3>
                <ProgressStats stats={stats} totalConcepts={concepts.length} compact />
              </div>

              {/* Category Progress */}
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  By Category
                </h3>
                <div className="space-y-4">
                  {categories.map((category) => {
                    const categoryProgress = getCategoryProgress(category.id);
                    return (
                      <div key={category.id} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-400">
                            {category.name}
                          </span>
                          <span className="text-gray-500 dark:text-gray-500">
                            {categoryProgress.learned}/{categoryProgress.total}
                          </span>
                        </div>
                        <ProgressBar
                          value={categoryProgress.learned}
                          max={categoryProgress.total || 1}
                          size="sm"
                          color="blue"
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Footer Actions */}
            <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 space-y-2">
              <div className="flex gap-2">
                <button
                  onClick={handleExport}
                  className={cn(
                    'flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg',
                    'bg-gray-100 dark:bg-gray-800',
                    'text-gray-700 dark:text-gray-300',
                    'hover:bg-gray-200 dark:hover:bg-gray-700',
                    'transition-colors text-sm'
                  )}
                >
                  <Download className="w-4 h-4" />
                  Export
                </button>
                <button
                  onClick={handleReset}
                  className={cn(
                    'flex items-center justify-center gap-2 px-4 py-2 rounded-lg',
                    'bg-red-100 dark:bg-red-900/30',
                    'text-red-700 dark:text-red-400',
                    'hover:bg-red-200 dark:hover:bg-red-900/50',
                    'transition-colors text-sm'
                  )}
                >
                  <RotateCcw className="w-4 h-4" />
                  Reset
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default ProgressPanel;
