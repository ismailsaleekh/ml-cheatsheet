/**
 * Progress Context - Provides progress state throughout the app
 */
import { createContext, useContext, type ReactNode } from 'react';
import { useProgress, type UseProgressResult } from '@/hooks/useProgress';

/**
 * Progress context
 */
const ProgressContext = createContext<UseProgressResult | null>(null);

/**
 * Progress provider props
 */
interface ProgressProviderProps {
  children: ReactNode;
}

/**
 * Progress provider component
 */
export function ProgressProvider({ children }: ProgressProviderProps) {
  const progress = useProgress();

  return (
    <ProgressContext.Provider value={progress}>
      {children}
    </ProgressContext.Provider>
  );
}

/**
 * Hook to access progress context
 */
export function useProgressContext(): UseProgressResult {
  const context = useContext(ProgressContext);
  if (!context) {
    throw new Error('useProgressContext must be used within a ProgressProvider');
  }
  return context;
}

export default ProgressContext;
