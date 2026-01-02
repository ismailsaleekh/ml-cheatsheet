/**
 * Custom hooks for working with concept data
 */
import { useMemo } from 'react';
import { useAppState } from '@/context/AppContext';
import type { Concept, Category, ConceptNode } from '@/types/concept';
import {
  getConceptById,
  getDirectChildren,
  getConceptPath,
  hasChildren,
  buildConceptTree,
  getCategoryForConcept,
  sortBySectionId,
} from '@/utils/dataHelpers';

/**
 * Hook to get a concept by ID
 */
export function useConcept(id: string | null): Concept | undefined {
  const { concepts } = useAppState();

  return useMemo(() => {
    if (!id) return undefined;
    return getConceptById(concepts, id);
  }, [concepts, id]);
}

/**
 * Hook to get children of a concept or category
 */
export function useConceptChildren(parentId: string | null): Concept[] {
  const { concepts } = useAppState();

  return useMemo(() => {
    if (!parentId) return [];
    const children = getDirectChildren(concepts, parentId);
    return sortBySectionId(children);
  }, [concepts, parentId]);
}

/**
 * Hook to check if a concept has children
 */
export function useHasChildren(conceptId: string): boolean {
  const { concepts } = useAppState();

  return useMemo(() => {
    return hasChildren(concepts, conceptId);
  }, [concepts, conceptId]);
}

/**
 * Hook to get the breadcrumb path for a concept
 */
export function useConceptPath(conceptId: string | null): Concept[] {
  const { concepts } = useAppState();

  return useMemo(() => {
    if (!conceptId) return [];
    return getConceptPath(concepts, conceptId);
  }, [concepts, conceptId]);
}

/**
 * Hook to get the category for a concept
 */
export function useConceptCategory(conceptId: string | null): Category | undefined {
  const { concepts, categories } = useAppState();

  return useMemo(() => {
    if (!conceptId) return undefined;
    return getCategoryForConcept(concepts, categories, conceptId);
  }, [concepts, categories, conceptId]);
}

/**
 * Hook to get all categories
 */
export function useCategories(): Category[] {
  const { categories } = useAppState();

  return useMemo(() => {
    return [...categories].sort((a, b) => a.order - b.order);
  }, [categories]);
}

/**
 * Hook to get concepts grouped by category
 */
export function useConceptsByCategory(): Map<string, Concept[]> {
  const { concepts, categories } = useAppState();

  return useMemo(() => {
    const grouped = new Map<string, Concept[]>();

    categories.forEach((category) => {
      const categoryChildren = getDirectChildren(concepts, category.id);
      grouped.set(category.id, sortBySectionId(categoryChildren));
    });

    return grouped;
  }, [concepts, categories]);
}

/**
 * Hook to build the concept tree for a category
 */
export function useCategoryTree(categoryId: string): ConceptNode[] {
  const { concepts } = useAppState();

  return useMemo(() => {
    return buildConceptTree(concepts, categoryId);
  }, [concepts, categoryId]);
}

/**
 * Hook to get the full concept tree
 */
export function useConceptTree(): Map<string, ConceptNode[]> {
  const { concepts, categories } = useAppState();

  return useMemo(() => {
    const trees = new Map<string, ConceptNode[]>();

    categories.forEach((category) => {
      trees.set(category.id, buildConceptTree(concepts, category.id));
    });

    return trees;
  }, [concepts, categories]);
}

/**
 * Hook to get expanded state for a concept
 */
export function useIsExpanded(conceptId: string): boolean {
  const { expandedIds } = useAppState();
  return expandedIds.has(conceptId);
}

/**
 * Hook to get the selected concept
 */
export function useSelectedConcept(): Concept | undefined {
  const { concepts, selectedConceptId } = useAppState();

  return useMemo(() => {
    if (!selectedConceptId) return undefined;
    return getConceptById(concepts, selectedConceptId);
  }, [concepts, selectedConceptId]);
}

/**
 * Hook to get root-level items (categories with their direct children)
 */
export function useRootItems(): Array<{ category: Category; children: Concept[] }> {
  const categories = useCategories();
  const conceptsByCategory = useConceptsByCategory();

  return useMemo(() => {
    return categories.map((category) => ({
      category,
      children: conceptsByCategory.get(category.id) || [],
    }));
  }, [categories, conceptsByCategory]);
}

/**
 * Hook to get the loading and error state
 */
export function useDataStatus(): { isLoading: boolean; error: string | null } {
  const { isLoading, error } = useAppState();
  return { isLoading, error };
}
