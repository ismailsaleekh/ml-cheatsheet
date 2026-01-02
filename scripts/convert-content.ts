/**
 * Convert markdown content files to JSON format
 * Run with: npx ts-node scripts/convert-content.ts
 */
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface Example {
  description: string;
  code?: string;
  codeLanguage?: string;
}

interface Concept {
  id: string;
  name: string;
  parentId: string;
  sectionId: string;
  level: number;
  fullExplanation: string;
  simpleExplanation: string;
  example: Example;
  tags: string[];
  relatedConcepts: string[];
  prerequisites: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

interface Category {
  id: string;
  name: string;
  description: string;
  icon: string;
  color: string;
  order: number;
}

// Category mapping
const CATEGORIES: Record<string, Category> = {
  '7': { id: 'supervised-regression', name: '7. Regression', description: 'Supervised learning for continuous predictions', icon: 'TrendingUp', color: 'blue', order: 7 },
  '8': { id: 'supervised-classification', name: '8. Classification', description: 'Supervised learning for categorical predictions', icon: 'Tags', color: 'green', order: 8 },
  '9': { id: 'ensemble-methods', name: '9. Ensemble Methods', description: 'Combining multiple models for better predictions', icon: 'Users', color: 'purple', order: 9 },
  '10': { id: 'neural-networks', name: '10. Neural Networks', description: 'Foundations of deep learning', icon: 'Cpu', color: 'orange', order: 10 },
  '11': { id: 'deep-learning', name: '11. Deep Learning', description: 'Advanced neural network architectures', icon: 'Layers', color: 'red', order: 11 },
  '12': { id: 'unsupervised-learning', name: '12. Unsupervised Learning', description: 'Learning patterns from unlabeled data', icon: 'GitBranch', color: 'teal', order: 12 },
  '13': { id: 'generative-models', name: '13. Generative Models', description: 'Models that create new data', icon: 'Sparkles', color: 'pink', order: 13 },
  '14': { id: 'specialized-learning', name: '14. Specialized Learning', description: 'Transfer, reinforcement, and self-supervised learning', icon: 'Zap', color: 'yellow', order: 14 },
  '15': { id: 'structured-prediction', name: '15. Structured Prediction', description: 'Sequence labeling and structured outputs', icon: 'GitMerge', color: 'indigo', order: 15 },
  '16': { id: 'computer-vision', name: '16. Computer Vision', description: 'Image and video understanding', icon: 'Eye', color: 'cyan', order: 16 },
  '17': { id: 'nlp', name: '17. NLP', description: 'Natural language processing', icon: 'MessageSquare', color: 'emerald', order: 17 },
  '18': { id: 'time-series', name: '18. Time Series', description: 'Temporal data analysis and forecasting', icon: 'Clock', color: 'amber', order: 18 },
  '19': { id: 'recommendations', name: '19. Recommendations', description: 'Recommendation systems', icon: 'ThumbsUp', color: 'rose', order: 19 },
  '20': { id: 'practical', name: '20. Practical ML', description: 'Feature engineering and model debugging', icon: 'Wrench', color: 'slate', order: 20 },
  '21': { id: 'mlops', name: '21. MLOps', description: 'Deploying and monitoring ML systems', icon: 'Server', color: 'violet', order: 21 },
  '22': { id: 'ethics', name: '22. Ethics & Fairness', description: 'Responsible AI and compliance', icon: 'Scale', color: 'lime', order: 22 },
  '23': { id: 'rag', name: '23. RAG', description: 'Retrieval-Augmented Generation', icon: 'Search', color: 'fuchsia', order: 23 }
};

// Subsection mapping for intermediate level concepts
const SUBSECTION_NAMES: Record<string, string> = {};

function getCategoryId(sectionNum: string): string {
  return CATEGORIES[sectionNum]?.id || 'foundations';
}

function getDifficulty(sectionNum: number): 'beginner' | 'intermediate' | 'advanced' {
  if (sectionNum <= 8) return 'beginner';
  if (sectionNum <= 15) return 'intermediate';
  return 'advanced';
}

function generateTags(name: string, sectionId: string): string[] {
  const tags: string[] = [];
  const sectionNum = sectionId.split('.')[0];

  const sectionTags: Record<string, string[]> = {
    '7': ['regression', 'supervised'],
    '8': ['classification', 'supervised'],
    '9': ['ensemble', 'boosting', 'bagging'],
    '10': ['neural-networks', 'deep-learning'],
    '11': ['deep-learning', 'architectures'],
    '12': ['unsupervised', 'clustering'],
    '13': ['generative', 'generation'],
    '14': ['transfer-learning', 'reinforcement'],
    '15': ['sequence', 'structured'],
    '16': ['vision', 'images'],
    '17': ['nlp', 'text'],
    '18': ['time-series', 'forecasting'],
    '19': ['recommendations', 'collaborative'],
    '20': ['practical', 'engineering'],
    '21': ['mlops', 'deployment'],
    '22': ['ethics', 'fairness'],
    '23': ['rag', 'retrieval']
  };

  tags.push(...(sectionTags[sectionNum] || []));

  // Add name-based tags (words longer than 3 chars)
  const nameLower = name.toLowerCase();
  const words = nameLower.split(/[\s\-]+/).filter(w => w.length > 3 && !tags.includes(w));
  tags.push(...words.slice(0, 3));

  return [...new Set(tags)];
}

function parseConceptBlock(block: string): Concept | null {
  // Parse header: #### X.X.X Concept Name
  const headerMatch = block.match(/####\s+(\d+\.\d+(?:\.\d+)?)\s+(.+)/);
  if (!headerMatch) return null;

  const sectionId = headerMatch[1];
  const name = headerMatch[2].trim();
  const sectionParts = sectionId.split('.');
  const sectionNum = parseInt(sectionParts[0]);

  // Parse ID
  const idMatch = block.match(/\*\*ID:\*\*\s+`([^`]+)`/);
  const id = idMatch ? idMatch[1] : name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

  // Parse Parent
  const parentMatch = block.match(/\*\*Parent:\*\*\s+`([^`]+)`/);
  const parentRaw = parentMatch ? parentMatch[1] : '';

  // Determine parent ID
  let parentId: string;
  if (sectionParts.length === 2) {
    // This is a subsection (X.X) - parent is the category
    parentId = getCategoryId(sectionParts[0]);
  } else if (sectionParts.length === 3) {
    // This is a concept (X.X.X) - parent is the subsection or category
    // Check if there's a subsection parent
    const subsectionId = `${sectionParts[0]}.${sectionParts[1]}`;
    parentId = SUBSECTION_NAMES[subsectionId] || getCategoryId(sectionParts[0]);
  } else {
    parentId = getCategoryId(sectionParts[0]);
  }

  // Parse Full Explanation
  const fullMatch = block.match(/\*\*Full Explanation:\*\*\n([\s\S]*?)(?=\*\*Simple Explanation:\*\*)/);
  const fullExplanation = fullMatch ? fullMatch[1].trim() : '';

  // Parse Simple Explanation
  const simpleMatch = block.match(/\*\*Simple Explanation:\*\*\n([\s\S]*?)(?=\*\*Example:\*\*|---)/);
  const simpleExplanation = simpleMatch ? simpleMatch[1].trim() : '';

  // Parse Example
  const exampleMatch = block.match(/\*\*Example:\*\*\n([\s\S]*?)(?=---|$)/);
  let exampleText = exampleMatch ? exampleMatch[1].trim() : '';

  // Extract code from example
  const codeMatch = exampleText.match(/```(\w+)?\n([\s\S]*?)```/);

  const example: Example = {
    description: codeMatch ? exampleText.replace(codeMatch[0], '').trim() : exampleText,
  };

  if (codeMatch) {
    example.code = codeMatch[2].trim();
    example.codeLanguage = codeMatch[1] || 'python';
  }

  // Determine level
  const level = sectionParts.length === 2 ? 1 : 2;

  // Store subsection name for later reference
  if (level === 1) {
    SUBSECTION_NAMES[sectionId] = id;
  }

  return {
    id,
    name,
    parentId,
    sectionId,
    level,
    fullExplanation,
    simpleExplanation,
    example,
    tags: generateTags(name, sectionId),
    relatedConcepts: [],
    prerequisites: [],
    difficulty: getDifficulty(sectionNum)
  };
}

function parseMarkdownFile(filePath: string): Concept[] {
  const content = fs.readFileSync(filePath, 'utf-8');
  const concepts: Concept[] = [];

  // First pass: find all subsections (### X.X Name) and register them
  const subsectionMatches = content.matchAll(/###\s+(\d+\.\d+)\s+([^\n]+)/g);
  for (const match of subsectionMatches) {
    const sectionId = match[1];
    const name = match[2].trim();
    const id = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    SUBSECTION_NAMES[sectionId] = id;

    // Create subsection concept
    const sectionNum = sectionId.split('.')[0];
    concepts.push({
      id,
      name,
      parentId: getCategoryId(sectionNum),
      sectionId,
      level: 1,
      fullExplanation: `${name} covers important concepts in this area of machine learning.`,
      simpleExplanation: `Learn about ${name.toLowerCase()} and related techniques.`,
      example: { description: `This section contains concepts related to ${name.toLowerCase()}.` },
      tags: generateTags(name, sectionId),
      relatedConcepts: [],
      prerequisites: [],
      difficulty: getDifficulty(parseInt(sectionNum))
    });
  }

  // Split by concept headers (#### X.X.X Title)
  const conceptBlocks = content.split(/(?=####\s+\d+\.\d+)/);

  for (const block of conceptBlocks) {
    if (!block.trim() || !block.match(/####\s+\d+\.\d+/)) continue;

    const concept = parseConceptBlock(block);
    if (concept) {
      concepts.push(concept);
    }
  }

  return concepts;
}

async function main() {
  const contentDir = path.join(__dirname, '../content');
  const outputPath = path.join(__dirname, '../public/data/ml-content.json');

  // Read existing content
  const existing = JSON.parse(fs.readFileSync(outputPath, 'utf-8'));

  // Get existing concept IDs to avoid duplicates
  const existingIds = new Set(existing.concepts.map((c: Concept) => c.id));

  // Add new categories
  const existingCategoryIds = new Set(existing.categories.map((c: Category) => c.id));
  for (const cat of Object.values(CATEGORIES)) {
    if (!existingCategoryIds.has(cat.id)) {
      existing.categories.push(cat);
    }
  }

  // Sort categories by order
  existing.categories.sort((a: Category, b: Category) => a.order - b.order);

  // Parse all markdown files
  const allConcepts: Concept[] = [];

  const files = fs.readdirSync(contentDir).filter(f => f.endsWith('.md') && f.startsWith('ml-content'));

  for (const file of files) {
    const filePath = path.join(contentDir, file);
    console.log(`Processing ${file}...`);

    // Clear subsection names between files
    Object.keys(SUBSECTION_NAMES).forEach(k => delete SUBSECTION_NAMES[k]);

    const concepts = parseMarkdownFile(filePath);

    // Filter out duplicates
    const newConcepts = concepts.filter(c => !existingIds.has(c.id));
    newConcepts.forEach(c => existingIds.add(c.id));

    allConcepts.push(...newConcepts);
    console.log(`  Found ${concepts.length} concepts, ${newConcepts.length} new`);
  }

  // Add new concepts
  existing.concepts.push(...allConcepts);
  existing.lastUpdated = new Date().toISOString().split('T')[0];

  // Write output
  fs.writeFileSync(outputPath, JSON.stringify(existing, null, 2));

  console.log(`\n=== Summary ===`);
  console.log(`New concepts added: ${allConcepts.length}`);
  console.log(`Total concepts: ${existing.concepts.length}`);
  console.log(`Total categories: ${existing.categories.length}`);
  console.log(`Output: ${outputPath}`);
}

main().catch(console.error);
