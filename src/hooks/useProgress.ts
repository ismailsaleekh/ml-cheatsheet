/**
 * Hook for managing learning progress and bookmarks
 */
import { useCallback, useMemo } from 'react';
import { useLocalStorage } from './useLocalStorage';
import { useAppState } from '@/context/AppContext';
import type {
  ProgressState,
  StudyStats,
  CategoryProgress,
  OverallProgress,
} from '@/types/progress';
import {
  DEFAULT_PROGRESS_STATE,
  MAX_RECENT_VISITED,
  PROGRESS_STORAGE_KEY,
} from '@/types/progress';

export interface UseProgressResult {
  // State
  learnedConcepts: Set<string>;
  bookmarkedConcepts: Set<string>;
  lastVisited: string[];
  stats: StudyStats;

  // Actions
  toggleLearned: (conceptId: string) => void;
  toggleBookmark: (conceptId: string) => void;
  recordVisit: (conceptId: string) => void;
  isLearned: (conceptId: string) => boolean;
  isBookmarked: (conceptId: string) => boolean;

  // Computed
  getCategoryProgress: (categoryId: string) => CategoryProgress;
  getOverallProgress: () => OverallProgress;

  // Export/Import
  exportProgress: () => string;
  importProgress: (data: string) => boolean;
  resetProgress: () => void;
}

/**
 * Get today's date as ISO string (date only)
 */
function getTodayDate(): string {
  return new Date().toISOString().split('T')[0];
}

/**
 * Calculate streak based on last study date
 */
function calculateStreak(
  lastDate: string | null,
  currentStreak: number,
  longestStreak: number
): { currentStreak: number; longestStreak: number } {
  const today = getTodayDate();

  if (!lastDate) {
    return { currentStreak: 1, longestStreak: Math.max(longestStreak, 1) };
  }

  const lastStudyDate = new Date(lastDate);
  const todayDate = new Date(today);
  const diffDays = Math.floor(
    (todayDate.getTime() - lastStudyDate.getTime()) / (1000 * 60 * 60 * 24)
  );

  if (diffDays === 0) {
    // Same day, no change
    return { currentStreak, longestStreak };
  } else if (diffDays === 1) {
    // Consecutive day
    const newStreak = currentStreak + 1;
    return { currentStreak: newStreak, longestStreak: Math.max(longestStreak, newStreak) };
  } else {
    // Streak broken
    return { currentStreak: 1, longestStreak };
  }
}

/**
 * Custom hook for progress tracking
 */
export function useProgress(): UseProgressResult {
  const { concepts, categories } = useAppState();
  const [progress, setProgress] = useLocalStorage<ProgressState>(
    PROGRESS_STORAGE_KEY,
    DEFAULT_PROGRESS_STATE
  );

  // Convert arrays to Sets for fast lookup
  const learnedConcepts = useMemo(
    () => new Set(progress.learnedConcepts),
    [progress.learnedConcepts]
  );

  const bookmarkedConcepts = useMemo(
    () => new Set(progress.bookmarkedConcepts),
    [progress.bookmarkedConcepts]
  );

  // Check if concept is learned
  const isLearned = useCallback(
    (conceptId: string) => learnedConcepts.has(conceptId),
    [learnedConcepts]
  );

  // Check if concept is bookmarked
  const isBookmarked = useCallback(
    (conceptId: string) => bookmarkedConcepts.has(conceptId),
    [bookmarkedConcepts]
  );

  // Toggle learned status
  const toggleLearned = useCallback(
    (conceptId: string) => {
      setProgress((prev) => {
        const isCurrentlyLearned = prev.learnedConcepts.includes(conceptId);
        const concept = concepts.find((c) => c.id === conceptId);
        const categoryId = concept?.parentId?.split('-')[0] || concept?.sectionId.split('.')[0];

        let newLearnedConcepts: string[];
        let categoryChange = 0;

        if (isCurrentlyLearned) {
          newLearnedConcepts = prev.learnedConcepts.filter((id) => id !== conceptId);
          categoryChange = -1;
        } else {
          newLearnedConcepts = [...prev.learnedConcepts, conceptId];
          categoryChange = 1;
        }

        // Update category progress
        const newCategoryProgress = { ...prev.studyStats.categoryProgress };
        if (categoryId) {
          newCategoryProgress[categoryId] =
            (newCategoryProgress[categoryId] || 0) + categoryChange;
          if (newCategoryProgress[categoryId] < 0) {
            newCategoryProgress[categoryId] = 0;
          }
        }

        // Update streak
        const { currentStreak, longestStreak } = !isCurrentlyLearned
          ? calculateStreak(
              prev.studyStats.lastStudyDate,
              prev.studyStats.currentStreak,
              prev.studyStats.longestStreak
            )
          : { currentStreak: prev.studyStats.currentStreak, longestStreak: prev.studyStats.longestStreak };

        return {
          ...prev,
          learnedConcepts: newLearnedConcepts,
          studyStats: {
            ...prev.studyStats,
            totalLearned: newLearnedConcepts.length,
            currentStreak,
            longestStreak,
            lastStudyDate: !isCurrentlyLearned ? getTodayDate() : prev.studyStats.lastStudyDate,
            categoryProgress: newCategoryProgress,
          },
        };
      });
    },
    [setProgress, concepts]
  );

  // Toggle bookmark status
  const toggleBookmark = useCallback(
    (conceptId: string) => {
      setProgress((prev) => {
        const isCurrentlyBookmarked = prev.bookmarkedConcepts.includes(conceptId);
        return {
          ...prev,
          bookmarkedConcepts: isCurrentlyBookmarked
            ? prev.bookmarkedConcepts.filter((id) => id !== conceptId)
            : [...prev.bookmarkedConcepts, conceptId],
        };
      });
    },
    [setProgress]
  );

  // Record a visit to a concept
  const recordVisit = useCallback(
    (conceptId: string) => {
      setProgress((prev) => {
        const filteredVisited = prev.lastVisited.filter((id) => id !== conceptId);
        const newVisited = [conceptId, ...filteredVisited].slice(0, MAX_RECENT_VISITED);
        return {
          ...prev,
          lastVisited: newVisited,
        };
      });
    },
    [setProgress]
  );

  // Get progress for a specific category
  const getCategoryProgress = useCallback(
    (categoryId: string): CategoryProgress => {
      const categoryConcepts = concepts.filter((c) => {
        const cCategory = c.sectionId.split('.')[0];
        const category = categories.find((cat) => cat.order.toString() === cCategory);
        return category?.id === categoryId;
      });

      const total = categoryConcepts.length;
      const learned = categoryConcepts.filter((c) => learnedConcepts.has(c.id)).length;
      const percentage = total > 0 ? Math.round((learned / total) * 100) : 0;

      return { categoryId, learned, total, percentage };
    },
    [concepts, categories, learnedConcepts]
  );

  // Get overall progress
  const getOverallProgress = useCallback((): OverallProgress => {
    const total = concepts.length;
    const learned = learnedConcepts.size;
    const percentage = total > 0 ? Math.round((learned / total) * 100) : 0;

    return { learned, total, percentage };
  }, [concepts.length, learnedConcepts.size]);

  // Export progress as JSON string
  const exportProgress = useCallback((): string => {
    return JSON.stringify(progress, null, 2);
  }, [progress]);

  // Import progress from JSON string
  const importProgress = useCallback(
    (data: string): boolean => {
      try {
        const imported = JSON.parse(data) as ProgressState;
        // Validate structure
        if (
          !Array.isArray(imported.learnedConcepts) ||
          !Array.isArray(imported.bookmarkedConcepts)
        ) {
          return false;
        }
        setProgress(imported);
        return true;
      } catch {
        return false;
      }
    },
    [setProgress]
  );

  // Reset all progress
  const resetProgress = useCallback(() => {
    setProgress(DEFAULT_PROGRESS_STATE);
  }, [setProgress]);

  return {
    learnedConcepts,
    bookmarkedConcepts,
    lastVisited: progress.lastVisited,
    stats: progress.studyStats,
    toggleLearned,
    toggleBookmark,
    recordVisit,
    isLearned,
    isBookmarked,
    getCategoryProgress,
    getOverallProgress,
    exportProgress,
    importProgress,
    resetProgress,
  };
}

export default useProgress;
