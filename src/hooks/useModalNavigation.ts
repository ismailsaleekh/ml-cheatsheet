/**
 * Hook for modal navigation (previous/next concept)
 */
import { useMemo, useCallback } from 'react';
import { useAppState, useAppDispatch } from '@/context/AppContext';
import { getConceptById, getDirectChildren, sortBySectionId } from '@/utils/dataHelpers';

interface ModalNavigationResult {
  previousConcept: { id: string; name: string } | null;
  nextConcept: { id: string; name: string } | null;
  goToPrevious: () => void;
  goToNext: () => void;
}

/**
 * Custom hook for navigating between sibling concepts in modal
 * @param currentConceptId - ID of the currently selected concept
 */
export function useModalNavigation(currentConceptId: string | null): ModalNavigationResult {
  const { concepts } = useAppState();
  const dispatch = useAppDispatch();

  // Find siblings (concepts with same parent)
  const siblings = useMemo(() => {
    if (!currentConceptId) return [];

    const current = getConceptById(concepts, currentConceptId);
    if (!current || !current.parentId) return [];

    const siblingConcepts = getDirectChildren(concepts, current.parentId);
    return sortBySectionId(siblingConcepts);
  }, [concepts, currentConceptId]);

  // Find current index in siblings
  const currentIndex = useMemo(() => {
    return siblings.findIndex((c) => c.id === currentConceptId);
  }, [siblings, currentConceptId]);

  // Previous and next concepts
  const previousConcept = useMemo(() => {
    if (currentIndex <= 0) return null;
    const prev = siblings[currentIndex - 1];
    return { id: prev.id, name: prev.name };
  }, [siblings, currentIndex]);

  const nextConcept = useMemo(() => {
    if (currentIndex < 0 || currentIndex >= siblings.length - 1) return null;
    const next = siblings[currentIndex + 1];
    return { id: next.id, name: next.name };
  }, [siblings, currentIndex]);

  // Navigation handlers
  const goToPrevious = useCallback(() => {
    if (previousConcept) {
      dispatch({ type: 'SELECT_CONCEPT', payload: previousConcept.id });
    }
  }, [dispatch, previousConcept]);

  const goToNext = useCallback(() => {
    if (nextConcept) {
      dispatch({ type: 'SELECT_CONCEPT', payload: nextConcept.id });
    }
  }, [dispatch, nextConcept]);

  return {
    previousConcept,
    nextConcept,
    goToPrevious,
    goToNext,
  };
}

export default useModalNavigation;
