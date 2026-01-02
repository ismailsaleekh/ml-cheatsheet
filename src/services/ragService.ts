/**
 * RAGService - Main RAG service combining embeddings, vector search, and LLM
 */
import type { Concept } from '@/types/concept';
import type {
  SemanticSearchResult,
  ChatMessage,
  RAGConfig,
  RAGStatus,
  RAGContext,
} from '@/types/rag';
import { DEFAULT_RAG_CONFIG } from '@/types/rag';
import { vectorStore } from './vectorStore';
import {
  generateEmbedding,
  loadPrecomputedEmbeddings,
} from './embeddingService';

/**
 * Path to pre-computed embeddings
 */
const EMBEDDINGS_URL = '/data/embeddings/concept-embeddings.json';

/**
 * RAG Service class
 */
class RAGServiceClass {
  private config: RAGConfig = DEFAULT_RAG_CONFIG;
  private concepts: Map<string, Concept> = new Map();
  private isInitialized: boolean = false;
  private isLoading: boolean = false;
  private error: string | null = null;

  /**
   * Initialize the RAG service
   */
  async initialize(
    concepts: Concept[],
    config: Partial<RAGConfig> = {}
  ): Promise<void> {
    if (this.isLoading) return;

    this.isLoading = true;
    this.error = null;

    try {
      // Store concepts for lookup
      this.concepts.clear();
      for (const concept of concepts) {
        this.concepts.set(concept.id, concept);
      }

      // Update config
      this.config = { ...DEFAULT_RAG_CONFIG, ...config };

      // Try to load pre-computed embeddings
      try {
        const embeddings = await loadPrecomputedEmbeddings(EMBEDDINGS_URL);
        await vectorStore.load(embeddings);
        this.isInitialized = true;
      } catch (e) {
        // Pre-computed embeddings not available
        console.log('Pre-computed embeddings not available, semantic search will be limited');
        this.isInitialized = false;
      }
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to initialize RAG';
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Check if the service is ready
   */
  isReady(): boolean {
    return this.isInitialized && vectorStore.loaded;
  }

  /**
   * Get current status
   */
  getStatus(): RAGStatus {
    return {
      isReady: this.isReady(),
      isLoading: this.isLoading,
      error: this.error,
      embeddingsLoaded: vectorStore.loaded,
      provider: this.config.provider,
    };
  }

  /**
   * Perform semantic search
   */
  async semanticSearch(
    query: string,
    topK: number = 10
  ): Promise<SemanticSearchResult[]> {
    if (!this.isReady()) {
      return [];
    }

    try {
      // Generate query embedding
      const queryEmbedding = await generateEmbedding(query);

      // Search vector store
      const results = vectorStore.search(queryEmbedding, topK, 0.3);

      // Map to SemanticSearchResult
      return results
        .map((r) => {
          const concept = this.concepts.get(r.id);
          if (!concept) return null;
          return {
            conceptId: r.id,
            score: r.score,
            concept,
          };
        })
        .filter((r): r is SemanticSearchResult => r !== null);
    } catch (error) {
      console.error('Semantic search error:', error);
      return [];
    }
  }

  /**
   * Get relevant concepts for a query (for context building)
   */
  async getRelevantConcepts(
    query: string,
    topK: number = 5
  ): Promise<Concept[]> {
    const results = await this.semanticSearch(query, topK);
    return results.map((r) => r.concept);
  }

  /**
   * Find concepts similar to a given concept
   */
  async findSimilarConcepts(
    conceptId: string,
    topK: number = 5
  ): Promise<SemanticSearchResult[]> {
    if (!vectorStore.loaded) {
      return [];
    }

    try {
      const results = vectorStore.findNearest(conceptId, topK);
      return results
        .map((r) => {
          const concept = this.concepts.get(r.id);
          if (!concept) return null;
          return {
            conceptId: r.id,
            score: r.score,
            concept,
          };
        })
        .filter((r): r is SemanticSearchResult => r !== null);
    } catch (error) {
      console.error('Find similar concepts error:', error);
      return [];
    }
  }

  /**
   * Build context for chat from relevant concepts
   */
  buildContext(concepts: Concept[]): string {
    const contextParts = concepts.map((c, i) => {
      return `[${i + 1}] ${c.name}: ${c.simpleExplanation}`;
    });

    return `Relevant ML concepts:\n${contextParts.join('\n\n')}`;
  }

  /**
   * Chat with the RAG system (requires OpenAI API key)
   * This is a placeholder - full implementation requires OpenAI integration
   */
  async chat(
    _messages: ChatMessage[],
    _context?: RAGContext
  ): Promise<string> {
    if (this.config.provider !== 'openai' || !this.config.apiKey) {
      return 'AI chat requires an OpenAI API key. Please configure it in settings.';
    }

    // This would integrate with OpenAI API
    // For now, return a placeholder
    return 'OpenAI integration not yet implemented. Semantic search is available.';
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<RAGConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): RAGConfig {
    return { ...this.config };
  }

  /**
   * Reset the service
   */
  reset(): void {
    vectorStore.clear();
    this.concepts.clear();
    this.isInitialized = false;
    this.error = null;
  }
}

// Singleton instance
export const ragService = new RAGServiceClass();

export default ragService;
