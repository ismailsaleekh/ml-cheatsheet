# ML Cheatsheet

An interactive machine learning reference application with 374 concepts, semantic search, and RAG integration.

## Features

- **374 ML Concepts** across 23 categories covering foundations to advanced topics
- **Semantic Search** - Find concepts by meaning, not just keywords
- **Progress Tracking** - Track learned concepts with streaks and statistics
- **Bookmarks** - Save concepts for quick access
- **AI Chat Interface** - RAG-ready chat for asking questions about ML concepts
- **Dark/Light Mode** - System-aware theme switching
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Keyboard Shortcuts** - Power user navigation

## Categories

| # | Category | Concepts |
|---|----------|----------|
| 1-6 | Foundations, Data, Learning, Optimization, Regularization, Evaluation | 55 |
| 7-8 | Regression & Classification | 41 |
| 9 | Ensemble Methods | 24 |
| 10-11 | Neural Networks & Deep Learning | 50 |
| 12 | Unsupervised Learning | 21 |
| 13 | Generative Models | 26 |
| 14-15 | Specialized Learning & Structured Prediction | 21 |
| 16-17 | Computer Vision & NLP | 29 |
| 18-19 | Time Series & Recommendations | 23 |
| 20-22 | Practical ML, MLOps, Ethics | 28 |
| 23 | RAG (Retrieval-Augmented Generation) | 56 |

## Tech Stack

- **Frontend**: React 18, TypeScript
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Build Tool**: Vite
- **Embeddings**: Transformers.js (all-MiniLM-L6-v2)
- **Vector Search**: Custom in-memory vector store with cosine similarity

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Clone the repository
git clone https://github.com/ismailsaleekh/ml-cheatsheet.git
cd ml-cheatsheet

# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Build for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
ml-cheatsheet/
├── public/
│   └── data/
│       ├── ml-content.json          # All 374 concepts
│       └── embeddings/
│           └── concept-embeddings.json  # Vector embeddings
├── src/
│   ├── components/
│   │   ├── ai/          # AI chat interface
│   │   ├── blocks/      # Concept grid and cards
│   │   ├── bookmarks/   # Bookmark functionality
│   │   ├── common/      # Shared components
│   │   ├── detail/      # Concept detail modal
│   │   ├── layout/      # Header, main content
│   │   ├── progress/    # Progress tracking UI
│   │   └── search/      # Search components
│   ├── context/         # React contexts
│   ├── hooks/           # Custom hooks
│   ├── services/        # RAG, embeddings, vector store
│   ├── types/           # TypeScript types
│   └── utils/           # Helper functions
├── content/             # Markdown source files
└── scripts/             # Build scripts
```

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |
| `npx ts-node scripts/generate-embeddings.ts` | Regenerate embeddings |
| `npx ts-node scripts/convert-content.ts` | Convert markdown to JSON |

## Adding New Concepts

1. Add concepts to markdown files in `content/` following the existing format:

```markdown
#### 23.1.1 Concept Name

**ID:** `concept-id`
**Parent:** `23.1`

**Full Explanation:**
Detailed technical explanation...

**Simple Explanation:**
Beginner-friendly explanation...

**Example:**
Code examples or diagrams...
```

2. Run the conversion script:
```bash
npx ts-node scripts/convert-content.ts
```

3. Regenerate embeddings:
```bash
npx ts-node scripts/generate-embeddings.ts
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `/` or `Cmd+K` | Focus search |
| `Escape` | Close modal/search |
| `←` `→` | Navigate between concepts in modal |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Built with [Claude Code](https://claude.com/claude-code)
- Embeddings powered by [Transformers.js](https://huggingface.co/docs/transformers.js)
- Icons from [Lucide](https://lucide.dev/)
