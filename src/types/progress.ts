/**
 * Progress tracking type definitions
 */

/**
 * Study statistics for tracking learning progress
 */
export interface StudyStats {
  totalLearned: number;
  currentStreak: number;
  longestStreak: number;
  lastStudyDate: string | null;
  categoryProgress: Record<string, number>;
}

/**
 * Main progress state stored in localStorage
 */
export interface ProgressState {
  learnedConcepts: string[];
  bookmarkedConcepts: string[];
  lastVisited: string[];
  studyStats: StudyStats;
  version: string;
}

/**
 * Individual concept progress details
 */
export interface ConceptProgress {
  conceptId: string;
  learnedAt: string;
  visitCount: number;
  lastVisitedAt: string;
}

/**
 * Progress for a category
 */
export interface CategoryProgress {
  categoryId: string;
  learned: number;
  total: number;
  percentage: number;
}

/**
 * Overall progress summary
 */
export interface OverallProgress {
  learned: number;
  total: number;
  percentage: number;
}

/**
 * Default progress state
 */
export const DEFAULT_PROGRESS_STATE: ProgressState = {
  learnedConcepts: [],
  bookmarkedConcepts: [],
  lastVisited: [],
  studyStats: {
    totalLearned: 0,
    currentStreak: 0,
    longestStreak: 0,
    lastStudyDate: null,
    categoryProgress: {},
  },
  version: '1.0.0',
};

/**
 * Maximum number of recently visited concepts to track
 */
export const MAX_RECENT_VISITED = 20;

/**
 * LocalStorage key for progress data
 */
export const PROGRESS_STORAGE_KEY = 'ml-cheatsheet-progress';
