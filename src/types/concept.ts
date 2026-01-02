/**
 * Core concept type definitions for the ML Cheatsheet application
 */

// Table data for examples
export interface TableData {
  headers: string[];
  rows: string[][];
}

// Example structure for a concept
export interface Example {
  description: string;
  code?: string;
  codeLanguage?: 'python' | 'javascript' | 'pseudocode' | 'typescript';
  visual?: string;
  table?: TableData;
}

// Difficulty levels
export type Difficulty = 'beginner' | 'intermediate' | 'advanced';

// Category color options
export type CategoryColor =
  | 'blue'
  | 'green'
  | 'orange'
  | 'yellow'
  | 'purple'
  | 'teal'
  | 'indigo'
  | 'pink'
  | 'red'
  | 'gray'
  | 'amber';

// Main concept interface
export interface Concept {
  id: string;
  name: string;
  parentId: string | null;
  sectionId: string;
  level: number;
  fullExplanation: string;
  simpleExplanation: string;
  example: Example;
  tags: string[];
  relatedConcepts: string[];
  prerequisites: string[];
  difficulty: Difficulty;
}

// Category interface
export interface Category {
  id: string;
  name: string;
  description: string;
  icon: string;
  color: CategoryColor;
  order: number;
}

// Concept with computed children (for tree structure)
export interface ConceptNode extends Concept {
  children: ConceptNode[];
}

// Data file structure
export interface MLContentData {
  version: string;
  lastUpdated: string;
  categories: Category[];
  concepts: Concept[];
}

// Category color configuration
export interface CategoryColorConfig {
  bg: string;
  bgHover: string;
  bgLight: string;
  text: string;
  border: string;
  ring: string;
}

// Category colors mapping type
export type CategoryColorsMap = Record<string, CategoryColorConfig>;
