/**
 * Header component - Top navigation bar with search
 */
import { useState } from 'react';
import {
  BookOpen,
  Search,
  Sun,
  Moon,
  Expand,
  Minimize2,
  Monitor,
  BarChart3,
  MessageCircle,
} from 'lucide-react';
import { useAppState, useAppDispatch } from '@/context/AppContext';
import { useProgressContext } from '@/context/ProgressContext';
import { useMobileDetect } from '@/hooks/useMobileDetect';
import { cn } from '@/utils/cn';
import { SearchInput } from '@/components/search/SearchInput';
import { MobileSearchOverlay } from '@/components/search/MobileSearchOverlay';
import { BookmarkDropdown } from '@/components/bookmarks';
import { ProgressPanel } from '@/components/progress';
import { AIChatInterface } from '@/components/ai';

export const Header: React.FC = () => {
  const { theme } = useAppState();
  const dispatch = useAppDispatch();
  const { getOverallProgress } = useProgressContext();
  const { isMobile, isTablet } = useMobileDetect();
  const [mobileSearchOpen, setMobileSearchOpen] = useState(false);
  const [progressPanelOpen, setProgressPanelOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);

  const progress = getOverallProgress();

  const handleExpandAll = () => {
    dispatch({ type: 'EXPAND_ALL' });
  };

  const handleCollapseAll = () => {
    dispatch({ type: 'COLLAPSE_ALL' });
  };

  const handleThemeToggle = () => {
    const nextTheme = theme === 'light' ? 'dark' : theme === 'dark' ? 'system' : 'light';
    dispatch({ type: 'SET_THEME', payload: nextTheme });
  };

  const getThemeIcon = () => {
    switch (theme) {
      case 'light':
        return <Sun className="w-5 h-5" />;
      case 'dark':
        return <Moon className="w-5 h-5" />;
      default:
        return <Monitor className="w-5 h-5" />;
    }
  };

  const showMobileSearch = isMobile || isTablet;

  return (
    <>
      <header className="sticky top-0 z-40 w-full border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between gap-4">
            {/* Logo and Title */}
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500 text-white">
                <BookOpen className="w-5 h-5" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">
                  ML Cheatsheet
                </h1>
                <p className="text-xs text-gray-500 dark:text-gray-400 hidden sm:block">
                  Interactive Machine Learning Reference
                </p>
              </div>
            </div>

            {/* Desktop Search Bar */}
            {!showMobileSearch && (
              <div className="flex-1 max-w-md">
                <SearchInput placeholder="Search concepts..." />
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex items-center gap-1">
              {/* Mobile Search Button */}
              {showMobileSearch && (
                <button
                  onClick={() => setMobileSearchOpen(true)}
                  className={cn(
                    'p-2 rounded-lg transition-colors duration-200',
                    'text-gray-600 dark:text-gray-400',
                    'hover:bg-gray-100 dark:hover:bg-gray-800',
                    'focus:outline-none focus:ring-2 focus:ring-blue-500'
                  )}
                  title="Search concepts"
                  aria-label="Open search"
                >
                  <Search className="w-5 h-5" />
                </button>
              )}

              {/* AI Chat Button */}
              <button
                onClick={() => setChatOpen(true)}
                className={cn(
                  'p-2 rounded-lg transition-colors duration-200',
                  'text-gray-600 dark:text-gray-400',
                  'hover:bg-purple-100 dark:hover:bg-purple-900/30',
                  'hover:text-purple-600 dark:hover:text-purple-400',
                  'focus:outline-none focus:ring-2 focus:ring-purple-500'
                )}
                title="AI Chat"
                aria-label="Open AI chat"
              >
                <MessageCircle className="w-5 h-5" />
              </button>

              {/* Bookmarks */}
              <BookmarkDropdown />

              {/* Progress Button */}
              <button
                onClick={() => setProgressPanelOpen(true)}
                className={cn(
                  'relative p-2 rounded-lg transition-colors duration-200',
                  'text-gray-600 dark:text-gray-400',
                  'hover:bg-gray-100 dark:hover:bg-gray-800',
                  'focus:outline-none focus:ring-2 focus:ring-blue-500'
                )}
                title={`Progress: ${progress.percentage}%`}
                aria-label={`Progress: ${progress.learned} of ${progress.total} concepts learned`}
              >
                <BarChart3 className="w-5 h-5" />
                {progress.learned > 0 && (
                  <span className="absolute -top-0.5 -right-0.5 min-w-[16px] h-4 px-1 flex items-center justify-center text-[10px] font-medium text-white bg-green-500 rounded-full">
                    {progress.percentage}%
                  </span>
                )}
              </button>

              {/* Divider */}
              <div className="hidden sm:block w-px h-6 bg-gray-200 dark:bg-gray-700 mx-1" />

              {/* Expand All */}
              <button
                onClick={handleExpandAll}
                className={cn(
                  'hidden sm:flex p-2 rounded-lg transition-colors duration-200',
                  'text-gray-600 dark:text-gray-400',
                  'hover:bg-gray-100 dark:hover:bg-gray-800',
                  'focus:outline-none focus:ring-2 focus:ring-blue-500'
                )}
                title="Expand all sections"
                aria-label="Expand all sections"
              >
                <Expand className="w-5 h-5" />
              </button>

              {/* Collapse All */}
              <button
                onClick={handleCollapseAll}
                className={cn(
                  'hidden sm:flex p-2 rounded-lg transition-colors duration-200',
                  'text-gray-600 dark:text-gray-400',
                  'hover:bg-gray-100 dark:hover:bg-gray-800',
                  'focus:outline-none focus:ring-2 focus:ring-blue-500'
                )}
                title="Collapse all sections"
                aria-label="Collapse all sections"
              >
                <Minimize2 className="w-5 h-5" />
              </button>

              {/* Theme Toggle */}
              <button
                onClick={handleThemeToggle}
                className={cn(
                  'p-2 rounded-lg transition-colors duration-200',
                  'text-gray-600 dark:text-gray-400',
                  'hover:bg-gray-100 dark:hover:bg-gray-800',
                  'focus:outline-none focus:ring-2 focus:ring-blue-500'
                )}
                title={`Current theme: ${theme}. Click to change.`}
                aria-label={`Current theme: ${theme}. Click to change.`}
              >
                {getThemeIcon()}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Search Overlay */}
      <MobileSearchOverlay
        isOpen={mobileSearchOpen}
        onClose={() => setMobileSearchOpen(false)}
      />

      {/* Progress Panel */}
      <ProgressPanel
        isOpen={progressPanelOpen}
        onClose={() => setProgressPanelOpen(false)}
      />

      {/* AI Chat Interface */}
      <AIChatInterface
        isOpen={chatOpen}
        onClose={() => setChatOpen(false)}
      />
    </>
  );
};

export default Header;
