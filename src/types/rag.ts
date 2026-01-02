/**
 * RAG (Retrieval Augmented Generation) type definitions
 */
import type { Concept } from './concept';

/**
 * Embedding for a concept
 */
export interface ConceptEmbedding {
  conceptId: string;
  embedding: number[];
  textUsed: string;
}

/**
 * Result from semantic search
 */
export interface SemanticSearchResult {
  conceptId: string;
  score: number;
  concept: Concept;
}

/**
 * Chat message in AI conversation
 */
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  sources?: string[];
}

/**
 * RAG provider configuration
 */
export type RAGProvider = 'local' | 'openai';

/**
 * RAG configuration options
 */
export interface RAGConfig {
  provider: RAGProvider;
  apiKey?: string;
  modelId?: string;
  embeddingModel?: string;
  maxTokens?: number;
  temperature?: number;
}

/**
 * Context for RAG queries
 */
export interface RAGContext {
  concepts: Concept[];
  maxContextLength: number;
  systemPrompt: string;
}

/**
 * RAG service status
 */
export interface RAGStatus {
  isReady: boolean;
  isLoading: boolean;
  error: string | null;
  embeddingsLoaded: boolean;
  provider: RAGProvider;
}

/**
 * Search mode for toggling between keyword and semantic search
 */
export type SearchMode = 'keyword' | 'semantic';

/**
 * Default RAG configuration
 */
export const DEFAULT_RAG_CONFIG: RAGConfig = {
  provider: 'local',
  maxTokens: 1000,
  temperature: 0.7,
};

/**
 * System prompt for AI chat
 */
export const DEFAULT_SYSTEM_PROMPT = `You are an expert machine learning tutor helping users understand ML concepts.
You have access to a comprehensive ML cheatsheet with detailed explanations of concepts.
When answering questions:
1. Be concise but thorough
2. Use examples when helpful
3. Reference specific concepts from the cheatsheet
4. Explain at an appropriate level based on the concept's difficulty
5. If you reference a concept, mention its name so users can look it up`;

/**
 * LocalStorage key for RAG configuration
 */
export const RAG_CONFIG_STORAGE_KEY = 'ml-cheatsheet-rag-config';

/**
 * LocalStorage key for chat history
 */
export const CHAT_HISTORY_STORAGE_KEY = 'ml-cheatsheet-chat-history';
