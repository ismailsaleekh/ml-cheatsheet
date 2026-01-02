/**
 * Application state type definitions
 */

import type { Concept, Category } from './concept';

// Tab options for detail modal
export type DetailTab = 'technical' | 'simple' | 'example';

// Main application state
export interface AppState {
  // Data
  concepts: Concept[];
  categories: Category[];
  isLoading: boolean;
  error: string | null;

  // Navigation
  currentPath: string[];
  expandedIds: Set<string>;
  selectedConceptId: string | null;

  // Search
  searchQuery: string;
  searchResults: Concept[];
  isSearching: boolean;

  // UI
  sidebarOpen: boolean;
  modalOpen: boolean;
  activeTab: DetailTab;
  theme: 'light' | 'dark' | 'system';
}

// Action types for the reducer
export type AppAction =
  | { type: 'SET_DATA'; payload: { concepts: Concept[]; categories: Category[] } }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'TOGGLE_EXPAND'; payload: string }
  | { type: 'EXPAND_TO'; payload: string }
  | { type: 'EXPAND_MULTIPLE'; payload: string[] }
  | { type: 'SELECT_CONCEPT'; payload: string | null }
  | { type: 'SET_SEARCH'; payload: string }
  | { type: 'SET_SEARCH_RESULTS'; payload: Concept[] }
  | { type: 'CLEAR_SEARCH' }
  | { type: 'NAVIGATE_TO'; payload: string[] }
  | { type: 'SET_TAB'; payload: DetailTab }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'SET_MODAL_OPEN'; payload: boolean }
  | { type: 'EXPAND_ALL' }
  | { type: 'COLLAPSE_ALL' }
  | { type: 'SET_THEME'; payload: 'light' | 'dark' | 'system' };

// Initial state factory
export const createInitialState = (): AppState => ({
  // Data
  concepts: [],
  categories: [],
  isLoading: true,
  error: null,

  // Navigation
  currentPath: [],
  expandedIds: new Set<string>(),
  selectedConceptId: null,

  // Search
  searchQuery: '',
  searchResults: [],
  isSearching: false,

  // UI
  sidebarOpen: false,
  modalOpen: false,
  activeTab: 'technical',
  theme: 'system',
});

// App context value type
export interface AppContextValue {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
}
