/**
 * MainContent component - Main content area wrapper
 */
import React from 'react';
import { Loader2, AlertCircle } from 'lucide-react';
import { useDataStatus } from '@/hooks/useConceptData';
import { BlockContainer } from '@/components/blocks';
import { cn } from '@/utils/cn';

export const MainContent: React.FC = () => {
  const { isLoading, error } = useDataStatus();

  return (
    <main className="flex-1">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Loading State */}
        {isLoading && (
          <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
            <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
            <p className="text-gray-500 dark:text-gray-400">
              Loading ML concepts...
            </p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
            <div className="p-4 rounded-full bg-red-100 dark:bg-red-900/30">
              <AlertCircle className="w-8 h-8 text-red-500" />
            </div>
            <div className="text-center">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Failed to load data
              </h2>
              <p className="mt-1 text-gray-500 dark:text-gray-400">
                {error}
              </p>
              <button
                onClick={() => window.location.reload()}
                className={cn(
                  'mt-4 px-4 py-2 rounded-lg',
                  'bg-blue-500 text-white',
                  'hover:bg-blue-600',
                  'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2',
                  'transition-colors duration-200'
                )}
              >
                Try Again
              </button>
            </div>
          </div>
        )}

        {/* Content */}
        {!isLoading && !error && (
          <>
            {/* Page Title */}
            <div className="mb-8">
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Machine Learning Concepts
              </h1>
              <p className="mt-2 text-gray-500 dark:text-gray-400">
                Explore 50+ ML concepts organized by category. Click on any concept to expand
                and see related topics, or click the info icon for detailed explanations.
              </p>
            </div>

            {/* Stats Bar */}
            <div className="mb-8 flex flex-wrap gap-4">
              <StatBadge label="Categories" value="6" />
              <StatBadge label="Concepts" value="50+" />
              <StatBadge label="Levels" value="3" />
            </div>

            {/* Block Container */}
            <BlockContainer />
          </>
        )}
      </div>
    </main>
  );
};

/**
 * Small stat badge component
 */
interface StatBadgeProps {
  label: string;
  value: string;
}

const StatBadge: React.FC<StatBadgeProps> = ({ label, value }) => (
  <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-gray-100 dark:bg-gray-800">
    <span className="text-sm font-medium text-gray-900 dark:text-white">
      {value}
    </span>
    <span className="text-sm text-gray-500 dark:text-gray-400">
      {label}
    </span>
  </div>
);

export default MainContent;
