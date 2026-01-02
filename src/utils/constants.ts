/**
 * Application constants and configuration
 */
import type { CategoryColorConfig, CategoryColorsMap, Difficulty } from '@/types/concept';

/**
 * Category color configurations
 */
export const CATEGORY_COLORS: CategoryColorsMap = {
  foundations: {
    bg: 'bg-blue-500',
    bgHover: 'hover:bg-blue-600',
    bgLight: 'bg-blue-50 dark:bg-blue-950/30',
    text: 'text-blue-500 dark:text-blue-400',
    border: 'border-blue-500',
    ring: 'ring-blue-500',
  },
  'data-foundation': {
    bg: 'bg-green-500',
    bgHover: 'hover:bg-green-600',
    bgLight: 'bg-green-50 dark:bg-green-950/30',
    text: 'text-green-500 dark:text-green-400',
    border: 'border-green-500',
    ring: 'ring-green-500',
  },
  'learning-problem': {
    bg: 'bg-orange-500',
    bgHover: 'hover:bg-orange-600',
    bgLight: 'bg-orange-50 dark:bg-orange-950/30',
    text: 'text-orange-500 dark:text-orange-400',
    border: 'border-orange-500',
    ring: 'ring-orange-500',
  },
  optimization: {
    bg: 'bg-yellow-500',
    bgHover: 'hover:bg-yellow-600',
    bgLight: 'bg-yellow-50 dark:bg-yellow-950/30',
    text: 'text-yellow-600 dark:text-yellow-400',
    border: 'border-yellow-500',
    ring: 'ring-yellow-500',
  },
  regularization: {
    bg: 'bg-purple-500',
    bgHover: 'hover:bg-purple-600',
    bgLight: 'bg-purple-50 dark:bg-purple-950/30',
    text: 'text-purple-500 dark:text-purple-400',
    border: 'border-purple-500',
    ring: 'ring-purple-500',
  },
  evaluation: {
    bg: 'bg-teal-500',
    bgHover: 'hover:bg-teal-600',
    bgLight: 'bg-teal-50 dark:bg-teal-950/30',
    text: 'text-teal-500 dark:text-teal-400',
    border: 'border-teal-500',
    ring: 'ring-teal-500',
  },
  supervised: {
    bg: 'bg-indigo-500',
    bgHover: 'hover:bg-indigo-600',
    bgLight: 'bg-indigo-50 dark:bg-indigo-950/30',
    text: 'text-indigo-500 dark:text-indigo-400',
    border: 'border-indigo-500',
    ring: 'ring-indigo-500',
  },
  unsupervised: {
    bg: 'bg-pink-500',
    bgHover: 'hover:bg-pink-600',
    bgLight: 'bg-pink-50 dark:bg-pink-950/30',
    text: 'text-pink-500 dark:text-pink-400',
    border: 'border-pink-500',
    ring: 'ring-pink-500',
  },
  neural: {
    bg: 'bg-red-500',
    bgHover: 'hover:bg-red-600',
    bgLight: 'bg-red-50 dark:bg-red-950/30',
    text: 'text-red-500 dark:text-red-400',
    border: 'border-red-500',
    ring: 'ring-red-500',
  },
  mlops: {
    bg: 'bg-slate-500',
    bgHover: 'hover:bg-slate-600',
    bgLight: 'bg-slate-50 dark:bg-slate-950/30',
    text: 'text-slate-500 dark:text-slate-400',
    border: 'border-slate-500',
    ring: 'ring-slate-500',
  },
};

/**
 * Default category color for unknown categories
 */
export const DEFAULT_CATEGORY_COLOR: CategoryColorConfig = {
  bg: 'bg-gray-500',
  bgHover: 'hover:bg-gray-600',
  bgLight: 'bg-gray-50 dark:bg-gray-950/30',
  text: 'text-gray-500 dark:text-gray-400',
  border: 'border-gray-500',
  ring: 'ring-gray-500',
};

/**
 * Get category color config by category ID
 */
export function getCategoryColor(categoryId: string): CategoryColorConfig {
  return CATEGORY_COLORS[categoryId] || DEFAULT_CATEGORY_COLOR;
}

/**
 * Difficulty color configurations
 */
export const DIFFICULTY_COLORS: Record<Difficulty, string> = {
  beginner: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  intermediate: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  advanced: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
};

/**
 * Get difficulty badge classes
 */
export function getDifficultyColor(difficulty: Difficulty): string {
  return DIFFICULTY_COLORS[difficulty] || DIFFICULTY_COLORS.beginner;
}

/**
 * Animation variants for Framer Motion
 */
export const ANIMATION_VARIANTS = {
  expand: {
    initial: { opacity: 0, height: 0 },
    animate: { opacity: 1, height: 'auto' },
    exit: { opacity: 0, height: 0 },
    transition: { duration: 0.3, ease: 'easeOut' },
  },
  fadeIn: {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 },
    transition: { duration: 0.15 },
  },
  scaleIn: {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.95 },
    transition: { duration: 0.2 },
  },
  slideDown: {
    initial: { opacity: 0, y: -10 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -10 },
    transition: { duration: 0.2 },
  },
};

/**
 * Framer Motion container variants for staggered children
 */
export const STAGGER_CONTAINER = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.05,
    },
  },
};

export const STAGGER_ITEM = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
};

/**
 * Icon mapping for categories
 */
export const CATEGORY_ICONS: Record<string, string> = {
  foundations: 'BookOpen',
  'data-foundation': 'Database',
  'learning-problem': 'Target',
  optimization: 'TrendingUp',
  regularization: 'Shield',
  evaluation: 'BarChart',
  supervised: 'GitBranch',
  unsupervised: 'Layers',
  neural: 'Brain',
  mlops: 'Settings',
};

/**
 * Depth-based indentation (in pixels)
 */
export const DEPTH_INDENT = 24;

/**
 * Maximum depth to show in navigation
 */
export const MAX_DISPLAY_DEPTH = 5;

/**
 * Local storage keys
 */
export const STORAGE_KEYS = {
  THEME: 'ml-cheatsheet-theme',
  EXPANDED_IDS: 'ml-cheatsheet-expanded',
  PROGRESS: 'ml-cheatsheet-progress',
} as const;

/**
 * Search debounce delay in milliseconds
 */
export const SEARCH_DEBOUNCE_MS = 200;

/**
 * Maximum search results to display
 */
export const MAX_SEARCH_RESULTS = 20;
