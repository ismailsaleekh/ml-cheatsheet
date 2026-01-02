/**
 * DetailTabs - Tab navigation for modal content
 */
import { motion } from 'framer-motion';
import { BookOpen, MessageCircle, Code } from 'lucide-react';
import type { DetailTab } from '@/types/state';
import { cn } from '@/utils/cn';

interface DetailTabsProps {
  activeTab: DetailTab;
  onTabChange: (tab: DetailTab) => void;
}

const tabs: Array<{ id: DetailTab; label: string; icon: React.ReactNode; shortcut: string }> = [
  { id: 'technical', label: 'Technical', icon: <BookOpen className="w-4 h-4" />, shortcut: '1' },
  { id: 'simple', label: 'Simple', icon: <MessageCircle className="w-4 h-4" />, shortcut: '2' },
  { id: 'example', label: 'Example', icon: <Code className="w-4 h-4" />, shortcut: '3' },
];

export const DetailTabs: React.FC<DetailTabsProps> = ({ activeTab, onTabChange }) => {
  return (
    <div className="px-6 pt-4">
      <div
        className="flex border-b border-gray-200 dark:border-gray-700"
        role="tablist"
        aria-label="Content tabs"
      >
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              role="tab"
              aria-selected={isActive}
              aria-controls={`panel-${tab.id}`}
              id={`tab-${tab.id}`}
              onClick={() => onTabChange(tab.id)}
              className={cn(
                'relative flex items-center gap-2 px-4 py-3',
                'text-sm font-medium transition-colors duration-150',
                'focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500',
                isActive
                  ? 'text-blue-600 dark:text-blue-400'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
              )}
            >
              {tab.icon}
              <span>{tab.label}</span>
              <span className="hidden sm:inline-flex items-center justify-center w-5 h-5 ml-1 text-xs rounded bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400">
                {tab.shortcut}
              </span>

              {/* Active indicator */}
              {isActive && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500"
                  transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default DetailTabs;
