/**
 * EmbeddingService - Handles embedding generation using Transformers.js
 */
import type { ConceptEmbedding } from '@/types/rag';
import type { Concept } from '@/types/concept';

// Lazy load the transformers library
let pipeline: any = null;
let embedder: any = null;

/**
 * Model to use for embeddings
 * all-MiniLM-L6-v2 is a good balance of size (~23MB) and quality
 */
const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2';

/**
 * Initialize the embedding pipeline
 */
async function initializePipeline(): Promise<void> {
  if (embedder) return;

  try {
    // Dynamic import to avoid bundling transformers.js in initial load
    const transformers = await import('@xenova/transformers');
    pipeline = transformers.pipeline;

    embedder = await pipeline('feature-extraction', EMBEDDING_MODEL, {
      quantized: true, // Use quantized model for smaller size
    });
  } catch (error) {
    console.error('Failed to initialize embedding pipeline:', error);
    throw error;
  }
}

/**
 * Check if the embedding service is available
 */
export async function isEmbeddingServiceAvailable(): Promise<boolean> {
  try {
    await initializePipeline();
    return true;
  } catch {
    return false;
  }
}

/**
 * Generate embedding for a single text
 */
export async function generateEmbedding(text: string): Promise<number[]> {
  await initializePipeline();

  if (!embedder) {
    throw new Error('Embedding pipeline not initialized');
  }

  // Generate embedding
  const result = await embedder(text, {
    pooling: 'mean',
    normalize: true,
  });

  // Convert to regular array
  return Array.from(result.data);
}

/**
 * Generate embeddings for multiple texts
 */
export async function generateEmbeddings(texts: string[]): Promise<number[][]> {
  const embeddings: number[][] = [];

  for (const text of texts) {
    const embedding = await generateEmbedding(text);
    embeddings.push(embedding);
  }

  return embeddings;
}

/**
 * Create embedding text for a concept
 * Combines name, explanation, and tags for better semantic representation
 */
export function createConceptEmbeddingText(concept: Concept): string {
  const parts = [
    concept.name,
    concept.simpleExplanation,
    concept.tags.join(', '),
  ].filter(Boolean);

  return parts.join('. ');
}

/**
 * Generate embedding for a concept
 */
export async function generateConceptEmbedding(
  concept: Concept
): Promise<ConceptEmbedding> {
  const text = createConceptEmbeddingText(concept);
  const embedding = await generateEmbedding(text);

  return {
    conceptId: concept.id,
    embedding,
    textUsed: text.slice(0, 200),
  };
}

/**
 * Generate embeddings for all concepts
 * Shows progress via callback
 */
export async function generateAllConceptEmbeddings(
  concepts: Concept[],
  onProgress?: (current: number, total: number) => void
): Promise<ConceptEmbedding[]> {
  await initializePipeline();

  const embeddings: ConceptEmbedding[] = [];
  const total = concepts.length;

  for (let i = 0; i < total; i++) {
    const concept = concepts[i];
    const embedding = await generateConceptEmbedding(concept);
    embeddings.push(embedding);

    if (onProgress) {
      onProgress(i + 1, total);
    }
  }

  return embeddings;
}

/**
 * Load pre-computed embeddings from a URL
 */
export async function loadPrecomputedEmbeddings(
  url: string
): Promise<ConceptEmbedding[]> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load embeddings: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error loading pre-computed embeddings:', error);
    throw error;
  }
}

/**
 * Embedding service status
 */
export interface EmbeddingServiceStatus {
  isAvailable: boolean;
  isInitialized: boolean;
  modelName: string;
}

/**
 * Get embedding service status
 */
export function getEmbeddingServiceStatus(): EmbeddingServiceStatus {
  return {
    isAvailable: true,
    isInitialized: embedder !== null,
    modelName: EMBEDDING_MODEL,
  };
}

export default {
  generateEmbedding,
  generateEmbeddings,
  generateConceptEmbedding,
  generateAllConceptEmbeddings,
  loadPrecomputedEmbeddings,
  isEmbeddingServiceAvailable,
  getEmbeddingServiceStatus,
  createConceptEmbeddingText,
};
