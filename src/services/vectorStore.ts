/**
 * VectorStore - In-memory vector store for semantic search
 */
import type { ConceptEmbedding } from '@/types/rag';

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  if (magnitude === 0) return 0;

  return dotProduct / magnitude;
}

/**
 * Search result from vector store
 */
export interface VectorSearchResult {
  id: string;
  score: number;
}

/**
 * VectorStore class for managing embeddings and similarity search
 */
export class VectorStore {
  private embeddings: Map<string, number[]> = new Map();
  private dimension: number = 0;
  private isLoaded: boolean = false;

  /**
   * Load embeddings from an array of ConceptEmbedding objects
   */
  async load(embeddingsData: ConceptEmbedding[]): Promise<void> {
    this.embeddings.clear();

    for (const item of embeddingsData) {
      if (item.embedding && item.embedding.length > 0) {
        this.embeddings.set(item.conceptId, item.embedding);
        if (this.dimension === 0) {
          this.dimension = item.embedding.length;
        }
      }
    }

    this.isLoaded = this.embeddings.size > 0;
  }

  /**
   * Load embeddings from a URL (JSON file)
   */
  async loadFromUrl(url: string): Promise<void> {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch embeddings: ${response.status}`);
      }
      const data = await response.json();
      await this.load(data);
    } catch (error) {
      console.error('Error loading embeddings from URL:', error);
      throw error;
    }
  }

  /**
   * Check if the store has been loaded with embeddings
   */
  get loaded(): boolean {
    return this.isLoaded;
  }

  /**
   * Get the number of embeddings in the store
   */
  get size(): number {
    return this.embeddings.size;
  }

  /**
   * Get the embedding dimension
   */
  get embeddingDimension(): number {
    return this.dimension;
  }

  /**
   * Add a single embedding to the store
   */
  add(id: string, embedding: number[]): void {
    if (this.dimension === 0) {
      this.dimension = embedding.length;
    } else if (embedding.length !== this.dimension) {
      throw new Error(
        `Embedding dimension mismatch: expected ${this.dimension}, got ${embedding.length}`
      );
    }
    this.embeddings.set(id, embedding);
    this.isLoaded = true;
  }

  /**
   * Remove an embedding from the store
   */
  remove(id: string): boolean {
    return this.embeddings.delete(id);
  }

  /**
   * Get an embedding by ID
   */
  get(id: string): number[] | undefined {
    return this.embeddings.get(id);
  }

  /**
   * Check if an embedding exists
   */
  has(id: string): boolean {
    return this.embeddings.has(id);
  }

  /**
   * Search for similar embeddings
   * @param queryEmbedding - The query vector to search for
   * @param topK - Number of results to return
   * @param threshold - Minimum similarity score (0-1)
   */
  search(
    queryEmbedding: number[],
    topK: number = 10,
    threshold: number = 0
  ): VectorSearchResult[] {
    if (queryEmbedding.length !== this.dimension && this.dimension > 0) {
      throw new Error(
        `Query dimension mismatch: expected ${this.dimension}, got ${queryEmbedding.length}`
      );
    }

    const results: VectorSearchResult[] = [];

    for (const [id, embedding] of this.embeddings) {
      const score = cosineSimilarity(queryEmbedding, embedding);
      if (score >= threshold) {
        results.push({ id, score });
      }
    }

    // Sort by score descending and return top K
    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  /**
   * Find the K nearest neighbors to a given embedding ID
   */
  findNearest(id: string, topK: number = 5): VectorSearchResult[] {
    const embedding = this.embeddings.get(id);
    if (!embedding) {
      return [];
    }

    const results = this.search(embedding, topK + 1);
    // Filter out the query itself
    return results.filter((r) => r.id !== id).slice(0, topK);
  }

  /**
   * Clear all embeddings
   */
  clear(): void {
    this.embeddings.clear();
    this.dimension = 0;
    this.isLoaded = false;
  }

  /**
   * Export embeddings as an array
   */
  export(): ConceptEmbedding[] {
    const result: ConceptEmbedding[] = [];
    for (const [conceptId, embedding] of this.embeddings) {
      result.push({
        conceptId,
        embedding,
        textUsed: '',
      });
    }
    return result;
  }
}

// Singleton instance
export const vectorStore = new VectorStore();

export default VectorStore;
