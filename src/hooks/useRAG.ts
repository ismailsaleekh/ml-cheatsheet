/**
 * Hook for RAG functionality
 */
import { useState, useCallback, useEffect } from 'react';
import { useAppState } from '@/context/AppContext';
import { ragService } from '@/services/ragService';
import type { RAGStatus, RAGConfig, SemanticSearchResult } from '@/types/rag';

export interface UseRAGResult {
  // Status
  status: RAGStatus;
  isReady: boolean;
  isLoading: boolean;

  // Actions
  initialize: () => Promise<void>;
  semanticSearch: (query: string, topK?: number) => Promise<SemanticSearchResult[]>;
  findSimilar: (conceptId: string, topK?: number) => Promise<SemanticSearchResult[]>;
  updateConfig: (config: Partial<RAGConfig>) => void;

  // Config
  config: RAGConfig;
}

/**
 * Hook for using RAG service in components
 */
export function useRAG(): UseRAGResult {
  const { concepts } = useAppState();
  const [status, setStatus] = useState<RAGStatus>(ragService.getStatus());
  const [isLoading, setIsLoading] = useState(false);

  // Initialize RAG service when concepts are loaded
  const initialize = useCallback(async () => {
    if (concepts.length === 0 || isLoading) return;

    setIsLoading(true);
    try {
      await ragService.initialize(concepts);
      setStatus(ragService.getStatus());
    } catch (error) {
      console.error('Failed to initialize RAG:', error);
      setStatus(ragService.getStatus());
    } finally {
      setIsLoading(false);
    }
  }, [concepts, isLoading]);

  // Auto-initialize when concepts are available
  useEffect(() => {
    if (concepts.length > 0 && !ragService.isReady() && !isLoading) {
      initialize();
    }
  }, [concepts.length, initialize, isLoading]);

  // Semantic search
  const semanticSearch = useCallback(
    async (query: string, topK: number = 10): Promise<SemanticSearchResult[]> => {
      if (!ragService.isReady()) {
        return [];
      }
      return ragService.semanticSearch(query, topK);
    },
    []
  );

  // Find similar concepts
  const findSimilar = useCallback(
    async (conceptId: string, topK: number = 5): Promise<SemanticSearchResult[]> => {
      if (!ragService.isReady()) {
        return [];
      }
      return ragService.findSimilarConcepts(conceptId, topK);
    },
    []
  );

  // Update configuration
  const updateConfig = useCallback((config: Partial<RAGConfig>) => {
    ragService.updateConfig(config);
    setStatus(ragService.getStatus());
  }, []);

  return {
    status,
    isReady: ragService.isReady(),
    isLoading,
    initialize,
    semanticSearch,
    findSimilar,
    updateConfig,
    config: ragService.getConfig(),
  };
}

export default useRAG;
