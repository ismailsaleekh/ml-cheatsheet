# Content Conversion Plan: Markdown to JSON

## Overview

Convert 9 markdown content files (~200+ concepts) into the JSON format used by `ml-content.json` to expand the app from ~55 concepts to 350+ concepts.

---

## Source Files

| File | Sections | Est. Concepts |
|------|----------|---------------|
| `content/ml-content-sections-7-8.md` | Supervised Learning | ~35 |
| `content/ml-content-section-9.md` | Ensemble Methods | ~20 |
| `content/ml-content-sections-10-11.md` | Neural Networks & Deep Learning | ~40 |
| `content/ml-content-section-12.md` | Unsupervised Learning | ~20 |
| `content/ml-content-section-13.md` | Generative Models | ~25 |
| `content/ml-content-sections-14-15.md` | Specialized Learning | ~20 |
| `content/ml-content-sections-16-17.md` | Computer Vision & NLP | ~25 |
| `content/ml-content-sections-18-19.md` | Time Series & Recommendations | ~20 |
| `content/ml-content-sections-20-22.md` | Practical, MLOps, Ethics | ~25 |

**Total: ~230 new concepts**

---

## Target JSON Structure

### Category Object
```json
{
  "id": "category-id",
  "name": "7. Category Name",
  "description": "Brief description",
  "icon": "IconName",
  "color": "colorName",
  "order": 7
}
```

### Concept Object
```json
{
  "id": "concept-id",
  "name": "Concept Name",
  "parentId": "parent-concept-id or category-id",
  "sectionId": "7.1.1",
  "level": 1 | 2,
  "fullExplanation": "Detailed technical explanation...",
  "simpleExplanation": "Beginner-friendly explanation...",
  "example": {
    "description": "Example description or diagram...",
    "code": "optional code snippet",
    "codeLanguage": "python"
  },
  "tags": ["tag1", "tag2"],
  "relatedConcepts": ["related-id-1", "related-id-2"],
  "prerequisites": ["prereq-id"],
  "difficulty": "beginner" | "intermediate" | "advanced"
}
```

---

## Markdown Source Format

```markdown
#### 7.1.1 Concept Name

**ID:** `concept-id`
**Parent:** `7.1`

**Full Explanation:**
Technical explanation text...

**Simple Explanation:**
Beginner-friendly text...

**Example:**
Example with code blocks or diagrams...
```

---

## Conversion Steps

### Phase 1: Add New Categories

Add categories for sections 7-22 to the `categories` array:

```json
{
  "id": "supervised-regression",
  "name": "7. Regression",
  "description": "Supervised learning for continuous predictions",
  "icon": "TrendingUp",
  "color": "blue",
  "order": 7
},
{
  "id": "supervised-classification",
  "name": "8. Classification",
  "description": "Supervised learning for categorical predictions",
  "icon": "Tags",
  "color": "green",
  "order": 8
},
{
  "id": "ensemble-methods",
  "name": "9. Ensemble Methods",
  "description": "Combining multiple models for better predictions",
  "icon": "Users",
  "color": "purple",
  "order": 9
},
{
  "id": "neural-networks",
  "name": "10. Neural Networks",
  "description": "Foundations of deep learning",
  "icon": "Cpu",
  "color": "orange",
  "order": 10
},
{
  "id": "deep-learning",
  "name": "11. Deep Learning",
  "description": "Advanced neural network architectures",
  "icon": "Layers",
  "color": "red",
  "order": 11
},
{
  "id": "unsupervised-learning",
  "name": "12. Unsupervised Learning",
  "description": "Learning patterns from unlabeled data",
  "icon": "GitBranch",
  "color": "teal",
  "order": 12
},
{
  "id": "generative-models",
  "name": "13. Generative Models",
  "description": "Models that create new data",
  "icon": "Sparkles",
  "color": "pink",
  "order": 13
},
{
  "id": "specialized-learning",
  "name": "14. Specialized Learning",
  "description": "Transfer, reinforcement, and self-supervised learning",
  "icon": "Zap",
  "color": "yellow",
  "order": 14
},
{
  "id": "structured-prediction",
  "name": "15. Structured Prediction",
  "description": "Sequence labeling and structured outputs",
  "icon": "GitMerge",
  "color": "indigo",
  "order": 15
},
{
  "id": "computer-vision",
  "name": "16. Computer Vision",
  "description": "Image and video understanding",
  "icon": "Eye",
  "color": "cyan",
  "order": 16
},
{
  "id": "nlp",
  "name": "17. NLP",
  "description": "Natural language processing",
  "icon": "MessageSquare",
  "color": "emerald",
  "order": 17
},
{
  "id": "time-series",
  "name": "18. Time Series",
  "description": "Temporal data analysis and forecasting",
  "icon": "Clock",
  "color": "amber",
  "order": 18
},
{
  "id": "recommendations",
  "name": "19. Recommendations",
  "description": "Recommendation systems",
  "icon": "ThumbsUp",
  "color": "rose",
  "order": 19
},
{
  "id": "practical",
  "name": "20. Practical ML",
  "description": "Feature engineering and model debugging",
  "icon": "Wrench",
  "color": "slate",
  "order": 20
},
{
  "id": "mlops",
  "name": "21. MLOps",
  "description": "Deploying and monitoring ML systems",
  "icon": "Server",
  "color": "violet",
  "order": 21
},
{
  "id": "ethics",
  "name": "22. Ethics & Fairness",
  "description": "Responsible AI and compliance",
  "icon": "Scale",
  "color": "lime",
  "order": 22
},
{
  "id": "rag",
  "name": "23. RAG",
  "description": "Retrieval-Augmented Generation",
  "icon": "Search",
  "color": "fuchsia",
  "order": 23
}
```

### Phase 2: Create Conversion Script

Create `scripts/convert-content.ts`:

```typescript
/**
 * Convert markdown content files to JSON format
 */
import * as fs from 'fs';
import * as path from 'path';

interface Concept {
  id: string;
  name: string;
  parentId: string;
  sectionId: string;
  level: number;
  fullExplanation: string;
  simpleExplanation: string;
  example: {
    description: string;
    code?: string;
    codeLanguage?: string;
  };
  tags: string[];
  relatedConcepts: string[];
  prerequisites: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

function parseMarkdownFile(filePath: string): Concept[] {
  const content = fs.readFileSync(filePath, 'utf-8');
  const concepts: Concept[] = [];

  // Split by concept headers (#### X.X.X Title)
  const conceptBlocks = content.split(/(?=####\s+\d+\.\d+\.\d+)/);

  for (const block of conceptBlocks) {
    if (!block.trim()) continue;

    const concept = parseConceptBlock(block);
    if (concept) concepts.push(concept);
  }

  return concepts;
}

function parseConceptBlock(block: string): Concept | null {
  // Parse header: #### 7.1.1 Concept Name
  const headerMatch = block.match(/####\s+(\d+\.\d+\.\d+)\s+(.+)/);
  if (!headerMatch) return null;

  const sectionId = headerMatch[1];
  const name = headerMatch[2].trim();

  // Parse ID
  const idMatch = block.match(/\*\*ID:\*\*\s+`([^`]+)`/);
  const id = idMatch ? idMatch[1] : name.toLowerCase().replace(/\s+/g, '-');

  // Parse Parent
  const parentMatch = block.match(/\*\*Parent:\*\*\s+`([^`]+)`/);
  const parentId = parentMatch ? parentMatch[1] : '';

  // Parse Full Explanation
  const fullMatch = block.match(/\*\*Full Explanation:\*\*\n([\s\S]*?)(?=\*\*Simple Explanation:\*\*)/);
  const fullExplanation = fullMatch ? fullMatch[1].trim() : '';

  // Parse Simple Explanation
  const simpleMatch = block.match(/\*\*Simple Explanation:\*\*\n([\s\S]*?)(?=\*\*Example:\*\*)/);
  const simpleExplanation = simpleMatch ? simpleMatch[1].trim() : '';

  // Parse Example
  const exampleMatch = block.match(/\*\*Example:\*\*\n([\s\S]*?)(?=---|$)/);
  const exampleText = exampleMatch ? exampleMatch[1].trim() : '';

  // Extract code from example
  const codeMatch = exampleText.match(/```(\w+)?\n([\s\S]*?)```/);

  // Determine difficulty based on section
  const sectionNum = parseInt(sectionId.split('.')[0]);
  let difficulty: 'beginner' | 'intermediate' | 'advanced' = 'intermediate';
  if (sectionNum <= 8) difficulty = 'beginner';
  if (sectionNum >= 13) difficulty = 'advanced';

  // Determine level (1 for subsection headers, 2 for concepts)
  const level = sectionId.split('.').length === 2 ? 1 : 2;

  return {
    id,
    name,
    parentId: mapParentId(parentId, sectionId),
    sectionId,
    level,
    fullExplanation,
    simpleExplanation,
    example: {
      description: codeMatch ? exampleText.replace(codeMatch[0], '').trim() : exampleText,
      code: codeMatch ? codeMatch[2].trim() : undefined,
      codeLanguage: codeMatch ? (codeMatch[1] || 'python') : undefined
    },
    tags: generateTags(name, sectionId),
    relatedConcepts: [],
    prerequisites: [],
    difficulty
  };
}

function mapParentId(parent: string, sectionId: string): string {
  // Map section numbers to category IDs
  const categoryMap: Record<string, string> = {
    '7': 'supervised-regression',
    '8': 'supervised-classification',
    '9': 'ensemble-methods',
    '10': 'neural-networks',
    '11': 'deep-learning',
    '12': 'unsupervised-learning',
    '13': 'generative-models',
    '14': 'specialized-learning',
    '15': 'structured-prediction',
    '16': 'computer-vision',
    '17': 'nlp',
    '18': 'time-series',
    '19': 'recommendations',
    '20': 'practical',
    '21': 'mlops',
    '22': 'ethics',
    '23': 'rag'
  };

  const section = sectionId.split('.')[0];
  return categoryMap[section] || parent;
}

function generateTags(name: string, sectionId: string): string[] {
  const tags: string[] = [];
  const nameLower = name.toLowerCase();

  // Add section-based tags
  const section = sectionId.split('.')[0];
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

  tags.push(...(sectionTags[section] || []));

  // Add name-based tags
  const words = nameLower.split(/\s+/);
  tags.push(...words.filter(w => w.length > 3));

  return [...new Set(tags)];
}

// Main execution
async function main() {
  const contentDir = path.join(__dirname, '../content');
  const outputPath = path.join(__dirname, '../public/data/ml-content.json');

  // Read existing content
  const existing = JSON.parse(fs.readFileSync(outputPath, 'utf-8'));

  // Parse all markdown files
  const allConcepts: Concept[] = [];
  const files = fs.readdirSync(contentDir).filter(f => f.endsWith('.md'));

  for (const file of files) {
    const filePath = path.join(contentDir, file);
    const concepts = parseMarkdownFile(filePath);
    allConcepts.push(...concepts);
    console.log(`Parsed ${concepts.length} concepts from ${file}`);
  }

  // Merge with existing
  existing.concepts.push(...allConcepts);
  existing.lastUpdated = new Date().toISOString().split('T')[0];

  // Write output
  fs.writeFileSync(outputPath, JSON.stringify(existing, null, 2));
  console.log(`\nTotal concepts: ${existing.concepts.length}`);
}

main().catch(console.error);
```

### Phase 3: Run Conversion

```bash
# Install dependencies if needed
npm install -D @types/node

# Run conversion script
npx ts-node scripts/convert-content.ts

# Verify output
cat public/data/ml-content.json | jq '.concepts | length'
```

### Phase 4: Regenerate Embeddings

```bash
# Generate embeddings for new concepts
npx ts-node scripts/generate-embeddings.ts
```

### Phase 5: Verify & Test

1. **Start dev server**: `npm run dev`
2. **Check categories**: All 23 sections visible in sidebar
3. **Search test**: Search for concepts from new sections
4. **Semantic search**: Test RAG search finds new concepts
5. **Navigation**: Drill down into hierarchies

---

## Validation Checklist

- [ ] All 9 markdown files parsed
- [ ] ~230 new concepts added to JSON
- [ ] 17 new categories added (sections 7-23)
- [ ] Parent-child relationships correct
- [ ] Section IDs match hierarchy
- [ ] Examples include code blocks where available
- [ ] Tags generated for all concepts
- [ ] Embeddings regenerated
- [ ] App loads without errors
- [ ] Search finds new concepts
- [ ] Progress tracking works for new concepts

---

## Rollback Plan

If issues occur:
1. Restore `ml-content.json` from git: `git checkout public/data/ml-content.json`
2. Regenerate embeddings: `npx ts-node scripts/generate-embeddings.ts`
3. Debug conversion script
4. Re-run conversion

---

## Timeline

| Phase | Task | Status |
|-------|------|--------|
| 1 | Add categories to JSON | Pending |
| 2 | Create conversion script | Pending |
| 3 | Run conversion | Pending |
| 4 | Regenerate embeddings | Pending |
| 5 | Test & verify | Pending |

---

## Notes

- Markdown format is consistent across all content files
- Code examples in markdown use fenced code blocks
- Some concepts have diagrams in ASCII art (preserve as-is in description)
- RAG section (23) already in separate spec file, include in conversion
