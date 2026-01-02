/**
 * Navigation-related type definitions
 */

// Breadcrumb item for navigation trail
export interface BreadcrumbItem {
  id: string;
  name: string;
  level: number;
}

// Navigation state
export interface NavigationState {
  currentPath: string[];
  expandedIds: Set<string>;
  selectedConceptId: string | null;
}

// Navigation context value
export interface NavigationContextValue {
  currentPath: string[];
  expandedIds: Set<string>;
  selectedConceptId: string | null;
  navigateTo: (path: string[]) => void;
  toggleExpand: (id: string) => void;
  selectConcept: (id: string | null) => void;
  expandAll: () => void;
  collapseAll: () => void;
  isExpanded: (id: string) => boolean;
}
