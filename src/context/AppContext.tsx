/**
 * Main application context providing global state management
 */
import React, {
  createContext,
  useContext,
  useReducer,
  useEffect,
  useCallback,
  useMemo,
  type ReactNode,
} from 'react';
import type { AppState, AppAction, AppContextValue } from '@/types/state';
import { createInitialState } from '@/types/state';
import { loadMLContent, getAncestorIds } from '@/utils/dataHelpers';

/**
 * App reducer function
 */
function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_DATA':
      return {
        ...state,
        concepts: action.payload.concepts,
        categories: action.payload.categories,
        isLoading: false,
        error: null,
      };

    case 'SET_LOADING':
      return {
        ...state,
        isLoading: action.payload,
      };

    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        isLoading: false,
      };

    case 'TOGGLE_EXPAND': {
      const newExpanded = new Set(state.expandedIds);
      if (newExpanded.has(action.payload)) {
        newExpanded.delete(action.payload);
      } else {
        newExpanded.add(action.payload);
      }
      return {
        ...state,
        expandedIds: newExpanded,
      };
    }

    case 'EXPAND_TO': {
      // Expand all ancestors of a concept
      const ancestorIds = getAncestorIds(state.concepts, action.payload);
      const newExpanded = new Set(state.expandedIds);
      ancestorIds.forEach((id) => newExpanded.add(id));
      return {
        ...state,
        expandedIds: newExpanded,
      };
    }

    case 'EXPAND_MULTIPLE': {
      const newExpanded = new Set(state.expandedIds);
      action.payload.forEach((id) => newExpanded.add(id));
      return {
        ...state,
        expandedIds: newExpanded,
      };
    }

    case 'SELECT_CONCEPT':
      return {
        ...state,
        selectedConceptId: action.payload,
        modalOpen: action.payload !== null,
      };

    case 'SET_SEARCH':
      return {
        ...state,
        searchQuery: action.payload,
        isSearching: action.payload.length > 0,
      };

    case 'SET_SEARCH_RESULTS':
      return {
        ...state,
        searchResults: action.payload,
      };

    case 'CLEAR_SEARCH':
      return {
        ...state,
        searchQuery: '',
        searchResults: [],
        isSearching: false,
      };

    case 'NAVIGATE_TO':
      return {
        ...state,
        currentPath: action.payload,
      };

    case 'SET_TAB':
      return {
        ...state,
        activeTab: action.payload,
      };

    case 'TOGGLE_SIDEBAR':
      return {
        ...state,
        sidebarOpen: !state.sidebarOpen,
      };

    case 'SET_MODAL_OPEN':
      return {
        ...state,
        modalOpen: action.payload,
        selectedConceptId: action.payload ? state.selectedConceptId : null,
      };

    case 'EXPAND_ALL': {
      const allIds = new Set(state.concepts.map((c) => c.id));
      return {
        ...state,
        expandedIds: allIds,
      };
    }

    case 'COLLAPSE_ALL':
      return {
        ...state,
        expandedIds: new Set(),
      };

    case 'SET_THEME':
      return {
        ...state,
        theme: action.payload,
      };

    default:
      return state;
  }
}

/**
 * Create the context
 */
const AppContext = createContext<AppContextValue | null>(null);

/**
 * App Provider component
 */
interface AppProviderProps {
  children: ReactNode;
}

export function AppProvider({ children }: AppProviderProps) {
  const [state, dispatch] = useReducer(appReducer, undefined, createInitialState);

  // Load data on mount
  useEffect(() => {
    let mounted = true;

    async function fetchData() {
      try {
        dispatch({ type: 'SET_LOADING', payload: true });
        const data = await loadMLContent();

        if (mounted) {
          dispatch({
            type: 'SET_DATA',
            payload: {
              concepts: data.concepts,
              categories: data.categories,
            },
          });
        }
      } catch (error) {
        if (mounted) {
          dispatch({
            type: 'SET_ERROR',
            payload: error instanceof Error ? error.message : 'Failed to load data',
          });
        }
      }
    }

    fetchData();

    return () => {
      mounted = false;
    };
  }, []);

  // Apply theme
  useEffect(() => {
    const root = window.document.documentElement;

    if (state.theme === 'system') {
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light';
      root.classList.toggle('dark', systemTheme === 'dark');
    } else {
      root.classList.toggle('dark', state.theme === 'dark');
    }
  }, [state.theme]);

  const value = useMemo(
    () => ({
      state,
      dispatch,
    }),
    [state]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

/**
 * Hook to access the app context
 */
export function useAppContext(): AppContextValue {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
}

/**
 * Hook to access just the state
 */
export function useAppState(): AppState {
  const { state } = useAppContext();
  return state;
}

/**
 * Hook to access just the dispatch
 */
export function useAppDispatch(): React.Dispatch<AppAction> {
  const { dispatch } = useAppContext();
  return dispatch;
}

/**
 * Hook for concept operations
 */
export function useConceptActions() {
  const dispatch = useAppDispatch();
  const { concepts } = useAppState();

  const toggleExpand = useCallback(
    (id: string) => {
      dispatch({ type: 'TOGGLE_EXPAND', payload: id });
    },
    [dispatch]
  );

  const expandTo = useCallback(
    (id: string) => {
      dispatch({ type: 'EXPAND_TO', payload: id });
    },
    [dispatch]
  );

  const selectConcept = useCallback(
    (id: string | null) => {
      dispatch({ type: 'SELECT_CONCEPT', payload: id });
    },
    [dispatch]
  );

  const expandAll = useCallback(() => {
    dispatch({ type: 'EXPAND_ALL' });
  }, [dispatch]);

  const collapseAll = useCallback(() => {
    dispatch({ type: 'COLLAPSE_ALL' });
  }, [dispatch]);

  const isExpanded = useCallback(
    (id: string) => {
      const { expandedIds } = useAppState();
      return expandedIds.has(id);
    },
    [concepts]
  );

  return {
    toggleExpand,
    expandTo,
    selectConcept,
    expandAll,
    collapseAll,
    isExpanded,
  };
}
