/**
 * Script to generate embeddings for all concepts
 * Run with: npx ts-node scripts/generate-embeddings.ts
 */
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface Concept {
  id: string;
  name: string;
  simpleExplanation: string;
  tags: string[];
}

interface ConceptEmbedding {
  conceptId: string;
  embedding: number[];
  textUsed: string;
}

interface MLContentData {
  concepts: Concept[];
}

async function generateEmbeddings() {
  console.log('Loading concepts...');

  // Load concepts from JSON
  const contentPath = path.join(__dirname, '../public/data/ml-content.json');
  const contentData: MLContentData = JSON.parse(fs.readFileSync(contentPath, 'utf-8'));
  const concepts = contentData.concepts;

  console.log(`Found ${concepts.length} concepts`);

  // Import transformers dynamically
  console.log('Loading embedding model...');
  const { pipeline } = await import('@xenova/transformers');

  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    quantized: true,
  });

  console.log('Generating embeddings...');
  const embeddings: ConceptEmbedding[] = [];

  for (let i = 0; i < concepts.length; i++) {
    const concept = concepts[i];

    // Combine name, explanation, and tags for embedding
    const text = [
      concept.name,
      concept.simpleExplanation,
      concept.tags.join(', '),
    ]
      .filter(Boolean)
      .join('. ');

    // Generate embedding
    const result = await embedder(text, {
      pooling: 'mean',
      normalize: true,
    });

    embeddings.push({
      conceptId: concept.id,
      embedding: Array.from(result.data),
      textUsed: text.slice(0, 200),
    });

    // Progress
    if ((i + 1) % 10 === 0 || i === concepts.length - 1) {
      console.log(`Progress: ${i + 1}/${concepts.length}`);
    }
  }

  // Save embeddings
  const outputPath = path.join(__dirname, '../public/data/embeddings/concept-embeddings.json');
  const outputDir = path.dirname(outputPath);

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  fs.writeFileSync(outputPath, JSON.stringify(embeddings, null, 2));
  console.log(`Saved ${embeddings.length} embeddings to ${outputPath}`);

  // Also save a minified version
  const minifiedPath = outputPath.replace('.json', '.min.json');
  fs.writeFileSync(minifiedPath, JSON.stringify(embeddings));
  console.log(`Saved minified version to ${minifiedPath}`);
}

// Run
generateEmbeddings()
  .then(() => {
    console.log('Done!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('Error:', error);
    process.exit(1);
  });
