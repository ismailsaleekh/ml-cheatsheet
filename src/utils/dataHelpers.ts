/**
 * Data helper functions for working with concepts and categories
 */
import type { Concept, Category, ConceptNode, MLContentData } from '@/types/concept';

/**
 * Get a concept by its ID
 */
export function getConceptById(
  concepts: Concept[],
  id: string
): Concept | undefined {
  return concepts.find((c) => c.id === id);
}

/**
 * Get all child concepts for a given parent ID
 */
export function getChildConcepts(
  concepts: Concept[],
  parentId: string | null
): Concept[] {
  return concepts.filter((c) => c.parentId === parentId);
}

/**
 * Get the path from root to a specific concept (breadcrumb)
 */
export function getConceptPath(
  concepts: Concept[],
  conceptId: string
): Concept[] {
  const path: Concept[] = [];
  let currentId: string | null = conceptId;

  while (currentId) {
    const concept = getConceptById(concepts, currentId);
    if (concept) {
      path.unshift(concept);
      currentId = concept.parentId;
    } else {
      break;
    }
  }

  return path;
}

/**
 * Get all concepts belonging to a category (top-level parent is category)
 */
export function getConceptsByCategory(
  concepts: Concept[],
  categoryId: string
): Concept[] {
  return concepts.filter((c) => {
    // Direct children of category
    if (c.parentId === categoryId) return true;

    // Check if any ancestor is the category
    const path = getConceptPath(concepts, c.id);
    return path.some((p) => p.parentId === categoryId);
  });
}

/**
 * Get root-level concepts (direct children of categories)
 */
export function getRootConcepts(
  concepts: Concept[],
  categories: Category[]
): Concept[] {
  const categoryIds = new Set(categories.map((c) => c.id));
  return concepts.filter((c) => c.parentId && categoryIds.has(c.parentId));
}

/**
 * Get direct children of a concept or category
 */
export function getDirectChildren(
  concepts: Concept[],
  parentId: string
): Concept[] {
  return concepts.filter((c) => c.parentId === parentId);
}

/**
 * Check if a concept has children
 */
export function hasChildren(concepts: Concept[], conceptId: string): boolean {
  return concepts.some((c) => c.parentId === conceptId);
}

/**
 * Build a tree structure from flat concepts array
 */
export function buildConceptTree(
  concepts: Concept[],
  parentId: string | null = null
): ConceptNode[] {
  return concepts
    .filter((c) => c.parentId === parentId)
    .map((concept) => ({
      ...concept,
      children: buildConceptTree(concepts, concept.id),
    }));
}

/**
 * Get the category for a concept by traversing up the tree
 */
export function getCategoryForConcept(
  concepts: Concept[],
  categories: Category[],
  conceptId: string
): Category | undefined {
  const categoryIds = new Set(categories.map((c) => c.id));
  let currentId: string | null = conceptId;

  while (currentId) {
    const concept = getConceptById(concepts, currentId);
    if (!concept) break;

    if (concept.parentId && categoryIds.has(concept.parentId)) {
      return categories.find((c) => c.id === concept.parentId);
    }

    currentId = concept.parentId;
  }

  return undefined;
}

/**
 * Get the category ID for a concept
 */
export function getCategoryIdForConcept(
  concepts: Concept[],
  categories: Category[],
  conceptId: string
): string | undefined {
  const category = getCategoryForConcept(concepts, categories, conceptId);
  return category?.id;
}

/**
 * Flatten a concept tree into an array
 */
export function flattenConceptTree(nodes: ConceptNode[]): Concept[] {
  const result: Concept[] = [];

  function traverse(node: ConceptNode) {
    const { children, ...concept } = node;
    result.push(concept);
    children.forEach(traverse);
  }

  nodes.forEach(traverse);
  return result;
}

/**
 * Get all ancestor IDs for a concept
 */
export function getAncestorIds(
  concepts: Concept[],
  conceptId: string
): string[] {
  const path = getConceptPath(concepts, conceptId);
  return path.slice(0, -1).map((c) => c.id);
}

/**
 * Get all descendant IDs for a concept
 */
export function getDescendantIds(
  concepts: Concept[],
  conceptId: string
): string[] {
  const descendants: string[] = [];

  function collect(parentId: string) {
    const children = getDirectChildren(concepts, parentId);
    children.forEach((child) => {
      descendants.push(child.id);
      collect(child.id);
    });
  }

  collect(conceptId);
  return descendants;
}

/**
 * Count total concepts under a parent (including nested)
 */
export function countDescendants(
  concepts: Concept[],
  parentId: string
): number {
  return getDescendantIds(concepts, parentId).length;
}

/**
 * Load ML content data from JSON file
 */
export async function loadMLContent(): Promise<MLContentData> {
  const response = await fetch('/data/ml-content.json');
  if (!response.ok) {
    throw new Error(`Failed to load ML content: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Group concepts by their parent ID
 */
export function groupConceptsByParent(
  concepts: Concept[]
): Map<string | null, Concept[]> {
  const groups = new Map<string | null, Concept[]>();

  concepts.forEach((concept) => {
    const parentId = concept.parentId;
    if (!groups.has(parentId)) {
      groups.set(parentId, []);
    }
    groups.get(parentId)!.push(concept);
  });

  return groups;
}

/**
 * Sort concepts by their section ID
 */
export function sortBySectionId(concepts: Concept[]): Concept[] {
  return [...concepts].sort((a, b) => {
    const aParts = a.sectionId.split('.').map(Number);
    const bParts = b.sectionId.split('.').map(Number);

    for (let i = 0; i < Math.max(aParts.length, bParts.length); i++) {
      const aVal = aParts[i] ?? 0;
      const bVal = bParts[i] ?? 0;
      if (aVal !== bVal) return aVal - bVal;
    }

    return 0;
  });
}
