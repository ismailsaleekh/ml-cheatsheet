# RAG (Retrieval-Augmented Generation) - Complete Coverage

## Addition to ML Cheatsheet Content Specification

This section should be added as **Section 23: Retrieval-Augmented Generation (RAG)** in the main content specification.

---

## 23. RETRIEVAL-AUGMENTED GENERATION (RAG)

### Hierarchy Structure

```
23. RAG (Retrieval-Augmented Generation)
├── 23.1 RAG Fundamentals
│   ├── RAG Overview
│   ├── RAG vs Fine-Tuning
│   ├── RAG Architecture
│   ├── RAG Pipeline
│   ├── Grounding
│   └── Hallucination Mitigation
├── 23.2 Document Processing
│   ├── Document Loading
│   ├── Document Parsing
│   ├── Chunking
│   ├── Chunk Size
│   ├── Chunk Overlap
│   ├── Chunking Strategies
│   │   ├── Fixed-Size Chunking
│   │   ├── Semantic Chunking
│   │   ├── Recursive Chunking
│   │   ├── Document-Based Chunking
│   │   └── Agentic Chunking
│   └── Metadata Extraction
├── 23.3 Embeddings
│   ├── Text Embeddings
│   ├── Embedding Models
│   ├── Sentence Transformers
│   ├── OpenAI Embeddings
│   ├── Embedding Dimensions
│   ├── Semantic Similarity
│   ├── Cosine Similarity
│   ├── Dot Product Similarity
│   └── Embedding Fine-Tuning
├── 23.4 Vector Stores
│   ├── Vector Database
│   ├── Vector Index
│   ├── FAISS
│   ├── Pinecone
│   ├── Weaviate
│   ├── Chroma
│   ├── Milvus
│   ├── Qdrant
│   ├── pgvector
│   └── Index Types
│       ├── Flat Index
│       ├── IVF (Inverted File Index)
│       ├── HNSW
│       └── Product Quantization
├── 23.5 Retrieval Methods
│   ├── Retrieval Overview
│   ├── Dense Retrieval
│   ├── Sparse Retrieval
│   ├── BM25
│   ├── TF-IDF
│   ├── Hybrid Retrieval
│   ├── Top-K Retrieval
│   ├── Similarity Threshold
│   ├── Maximum Marginal Relevance (MMR)
│   └── Contextual Compression
├── 23.6 Query Processing
│   ├── Query Understanding
│   ├── Query Expansion
│   ├── Query Transformation
│   ├── Query Decomposition
│   ├── HyDE (Hypothetical Document Embeddings)
│   ├── Step-Back Prompting
│   ├── Multi-Query Retrieval
│   └── Query Routing
├── 23.7 Re-Ranking
│   ├── Re-Ranking Overview
│   ├── Cross-Encoder Re-Ranking
│   ├── Bi-Encoder vs Cross-Encoder
│   ├── Cohere Rerank
│   ├── ColBERT
│   ├── Lost in the Middle Problem
│   └── Reciprocal Rank Fusion (RRF)
├── 23.8 Context Management
│   ├── Context Window
│   ├── Context Stuffing
│   ├── Context Selection
│   ├── Context Ordering
│   ├── Token Limits
│   └── Long-Context Models
├── 23.9 Generation
│   ├── Prompt Construction
│   ├── System Prompt for RAG
│   ├── Context Injection
│   ├── Citation Generation
│   ├── Answer Synthesis
│   └── Faithfulness
├── 23.10 Advanced RAG Patterns
│   ├── Naive RAG
│   ├── Advanced RAG
│   ├── Modular RAG
│   ├── Self-RAG
│   ├── Corrective RAG (CRAG)
│   ├── Agentic RAG
│   ├── Graph RAG
│   ├── Multi-Modal RAG
│   ├── Conversational RAG
│   └── RAG Fusion
├── 23.11 RAG Evaluation
│   ├── RAG Evaluation Overview
│   ├── Retrieval Metrics
│   │   ├── Recall@K
│   │   ├── Precision@K
│   │   ├── MRR (Mean Reciprocal Rank)
│   │   ├── NDCG
│   │   └── Hit Rate
│   ├── Generation Metrics
│   │   ├── Faithfulness
│   │   ├── Answer Relevance
│   │   ├── Context Relevance
│   │   ├── Groundedness
│   │   └── Answer Correctness
│   ├── End-to-End Metrics
│   ├── RAGAS Framework
│   └── Human Evaluation
└── 23.12 RAG Infrastructure
    ├── RAG Frameworks
    │   ├── LangChain
    │   ├── LlamaIndex
    │   ├── Haystack
    │   └── Semantic Kernel
    ├── RAG in Production
    ├── Caching Strategies
    ├── Latency Optimization
    └── Cost Optimization
```

---

## DETAILED CONTENT

### 23.1 RAG Fundamentals

---

#### 23.1.1 RAG Overview

**ID:** `rag-overview`
**Parent:** `23.1`

**Full Explanation:**
Retrieval-Augmented Generation (RAG) is an architecture that enhances Large Language Model outputs by retrieving relevant information from external knowledge sources before generating responses. Instead of relying solely on knowledge encoded in model parameters during training, RAG dynamically fetches context-specific information at inference time, grounds responses in retrieved documents, and reduces hallucinations by providing factual anchors.

**Simple Explanation:**
Give the AI a reference book while it answers. Instead of relying only on what it memorized during training, RAG lets the AI look up relevant documents first, then answer based on what it found. Like an open-book exam instead of a closed-book one.

**Example:**
Without RAG:
- User: "What's our company's refund policy?"
- LLM: Makes up a generic policy (hallucination)

With RAG:
1. Query: "What's our company's refund policy?"
2. Retrieve: Find company_policy.pdf, section on refunds
3. Context: "Refunds are processed within 14 days for unused items..."
4. Generate: "According to your company policy, refunds are processed within 14 days for unused items..."

---

#### 23.1.2 RAG vs Fine-Tuning

**ID:** `rag-vs-finetuning`
**Parent:** `23.1`

**Full Explanation:**
RAG and fine-tuning are complementary approaches to customizing LLMs. Fine-tuning modifies model weights through training on domain-specific data, embedding knowledge permanently but requiring retraining for updates. RAG keeps the base model frozen and dynamically retrieves current information at inference time. RAG excels for frequently changing data and factual accuracy; fine-tuning excels for style, format, and reasoning patterns.

**Simple Explanation:**
Fine-tuning = Teaching the model new things permanently (surgery on its brain).
RAG = Giving the model reference materials to consult (giving it a handbook).

Fine-tuning is expensive and knowledge becomes outdated. RAG is cheaper and always uses current documents.

**Example:**
Company knowledge base with 10,000 documents:

Fine-tuning approach:
- Train model on all documents ($$$)
- New document added → Need to retrain
- Model might still hallucinate

RAG approach:
- Keep base model as-is
- Index documents in vector database
- Query retrieves relevant docs
- Model answers from retrieved context
- New document → Just add to index (instant)

| Aspect | Fine-Tuning | RAG |
|--------|-------------|-----|
| Update frequency | Requires retraining | Instant |
| Cost | High (training) | Lower (inference) |
| Factual accuracy | Can hallucinate | Grounded in sources |
| Best for | Style, format, reasoning | Facts, current info |

---

#### 23.1.3 RAG Architecture

**ID:** `rag-architecture`
**Parent:** `23.1`

**Full Explanation:**
The RAG architecture consists of two main components: a Retriever and a Generator. The Retriever converts queries and documents into embeddings, stores document embeddings in a vector database, and retrieves semantically similar documents given a query. The Generator (LLM) receives the original query plus retrieved context and synthesizes a grounded response. Components are connected through a pipeline orchestrating the flow.

**Simple Explanation:**
Two parts working together:
1. **Retriever** (the librarian): Finds relevant documents based on your question
2. **Generator** (the expert): Reads the documents and writes an answer

The retriever narrows down millions of documents to a handful of relevant ones; the generator crafts a response using them.

**Example:**
```
┌─────────────────────────────────────────────────────────────┐
│                      RAG ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐      ┌─────────────┐      ┌──────────────┐   │
│   │  Query  │ ──── │  Embedding  │ ──── │   Vector     │   │
│   │         │      │   Model     │      │   Database   │   │
│   └─────────┘      └─────────────┘      └──────┬───────┘   │
│                                                 │           │
│                                          Top-K Documents    │
│                                                 │           │
│   ┌─────────┐      ┌─────────────┐      ┌──────▼───────┐   │
│   │ Response│ ◄─── │     LLM     │ ◄─── │   Context    │   │
│   │         │      │  Generator  │      │  + Query     │   │
│   └─────────┘      └─────────────┘      └──────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

#### 23.1.4 RAG Pipeline

**ID:** `rag-pipeline`
**Parent:** `23.1`

**Full Explanation:**
The RAG pipeline consists of two phases: Indexing (offline) and Querying (online). Indexing: load documents → chunk into segments → generate embeddings → store in vector database. Querying: receive user query → embed query → retrieve similar chunks → (optionally) re-rank → construct prompt with context → generate response → (optionally) cite sources. Each stage can be customized and optimized independently.

**Simple Explanation:**
Two phases:
1. **Indexing (preparation)**: Break documents into pieces, convert to numbers, store for quick lookup
2. **Querying (live)**: Take question, find relevant pieces, give to LLM with question, get answer

Like organizing a library (indexing) then looking up books to answer questions (querying).

**Example:**
```
INDEXING PHASE (Once, offline):
Documents → Chunking → Embedding → Vector Store
   │            │           │            │
   ▼            ▼           ▼            ▼
[PDF,TXT]  [500 token   [768-dim     [Pinecone,
           chunks]      vectors]     Chroma]

QUERYING PHASE (Every request):
Query → Embed → Search → Retrieve → Rerank → Prompt → LLM → Response
  │        │       │         │         │        │       │        │
  ▼        ▼       ▼         ▼         ▼        ▼       ▼        ▼
"What    [768d]  Vector    Top-10   Top-3    Query  GPT-4  "The
is X?"          Store     chunks   best     +Ctx          answer..."
```

---

#### 23.1.5 Grounding

**ID:** `grounding`
**Parent:** `23.1`

**Full Explanation:**
Grounding refers to anchoring LLM outputs to verifiable information sources. In RAG, the model's response should be derivable from and consistent with the retrieved documents. Grounding improves factual accuracy, enables source attribution, and allows users to verify claims. Ungrounded responses (hallucinations) cannot be traced to any source.

**Simple Explanation:**
Making sure the AI's answer actually comes from real documents, not imagination. Every claim should be traceable to a source. If the AI says something, you can point to where it learned that.

**Example:**
Ungrounded (hallucination):
- "The company was founded in 1985" (made up)

Grounded:
- "According to the company website [Source 1], the company was founded in 1992"

Grounding check:
- Claim: "Founded in 1992"
- Source says: "Established in 1992"
- Grounded: ✓ Yes

---

#### 23.1.6 Hallucination Mitigation

**ID:** `hallucination-mitigation`
**Parent:** `23.1`

**Full Explanation:**
Hallucination in RAG occurs when the model generates information not supported by retrieved context or fabricates sources. Mitigation strategies include: explicit instructions to only use provided context, chain-of-thought prompting requiring source citation, confidence thresholds for uncertain answers, fact-checking against retrieved documents, and using models fine-tuned for faithfulness (like those trained with RLHF for attribution).

**Simple Explanation:**
Stop the AI from making stuff up. Strategies:
1. Tell it "only use provided documents"
2. Make it cite sources for every claim
3. Have it say "I don't know" when context doesn't contain the answer
4. Double-check its answers against the documents

**Example:**
Prompt engineering for hallucination mitigation:

```
System: You are a helpful assistant. Answer questions using ONLY 
the provided context. If the answer is not in the context, say 
"I don't have enough information to answer this question."
Do not make up information.

Context:
[Retrieved documents here]

User: What is the CEO's favorite color?

Bad response: "The CEO's favorite color is blue."
Good response: "I don't have enough information to answer this 
question. The provided documents don't mention the CEO's color 
preferences."
```

---

### 23.2 Document Processing

---

#### 23.2.1 Chunking

**ID:** `chunking`
**Parent:** `23.2`

**Full Explanation:**
Chunking divides documents into smaller segments suitable for embedding and retrieval. Chunks must be small enough to fit in context windows and contain focused information, but large enough to preserve meaning. Chunking strategy significantly impacts retrieval quality—poor chunking leads to incomplete context or irrelevant matches. Considerations include chunk size, overlap, boundary detection, and metadata preservation.

**Simple Explanation:**
Cutting documents into smaller pieces. You can't feed a 500-page book to the AI, so you slice it into digestible chunks. Too small = loses context; too big = too much noise. Finding the right size is crucial.

**Example:**
Original document (3000 words):
```
Chapter 1: Introduction
[1000 words about company history]

Chapter 2: Products  
[1000 words about product line]

Chapter 3: Pricing
[1000 words about pricing tiers]
```

Chunked (500 words each with overlap):
```
Chunk 1: Chapter 1 part 1 (words 1-500)
Chunk 2: Chapter 1 part 2 (words 400-900) ← 100 word overlap
Chunk 3: Chapter 1 part 3 + Chapter 2 start (words 800-1300)
... etc
```

---

#### 23.2.2 Chunk Size

**ID:** `chunk-size`
**Parent:** `23.2`

**Full Explanation:**
Chunk size determines the granularity of retrieval. Smaller chunks (100-200 tokens) provide precise retrieval but may lack context. Larger chunks (500-1000 tokens) preserve more context but may contain irrelevant information and retrieve less precisely. Optimal size depends on content type, query patterns, and embedding model capabilities. Empirical testing is essential—no universal optimal size exists.

**Simple Explanation:**
How big each piece should be. Small chunks = precise but might miss context. Big chunks = more context but might include irrelevant stuff. Usually 200-500 words works well, but test for your specific content.

**Example:**
Query: "What is the refund policy for electronics?"

Small chunks (100 tokens):
- Retrieves: "Electronics can be returned within 30 days"
- Missing: conditions, exceptions, process

Large chunks (1000 tokens):
- Retrieves: Full refund policy + unrelated warranty info + shipping policy
- Problem: Noise dilutes relevant information

Medium chunks (300 tokens):
- Retrieves: Complete refund policy section
- Just right: enough context, focused content

---

#### 23.2.3 Chunk Overlap

**ID:** `chunk-overlap`
**Parent:** `23.2`

**Full Explanation:**
Chunk overlap includes portions of adjacent chunks to prevent information loss at boundaries. Without overlap, a sentence split between two chunks loses coherence in both. Typical overlap is 10-20% of chunk size. Overlap improves retrieval recall but increases storage and computation. Critical for content where important information spans chunk boundaries.

**Simple Explanation:**
Let chunks share some text with their neighbors. If you cut right in the middle of an important sentence, both halves lose meaning. Overlap ensures nothing falls through the cracks between chunks.

**Example:**
Without overlap:
```
Chunk 1: "...The refund policy requires items to be"
Chunk 2: "returned in original packaging within 30 days."
```
Neither chunk has complete information!

With 50-word overlap:
```
Chunk 1: "...The refund policy requires items to be returned 
          in original packaging within 30 days."
Chunk 2: "returned in original packaging within 30 days. 
          Exceptions include..."
```
Both chunks contain the complete policy statement.

---

#### 23.2.4 Semantic Chunking

**ID:** `semantic-chunking`
**Parent:** `23.2`

**Full Explanation:**
Semantic chunking splits documents based on meaning rather than fixed character/token counts. It identifies natural boundaries using sentence embeddings—when semantic similarity between consecutive sentences drops significantly, a new chunk begins. This preserves coherent topics within chunks and avoids splitting related content. More expensive than fixed-size but often improves retrieval quality.

**Simple Explanation:**
Cut documents where topics naturally change, not at arbitrary word counts. Use AI to detect "this paragraph is about something different than the last one" and split there. Keeps related ideas together.

**Example:**
Document about a product:
```
[Paragraph 1-3: Product features]
[Paragraph 4-5: Pricing information]  
[Paragraph 6-8: Customer reviews]
```

Fixed-size chunking: Might split paragraph 3 and 4 together (mixing features and pricing)

Semantic chunking:
- Chunk 1: Paragraphs 1-3 (all features)
- Chunk 2: Paragraphs 4-5 (all pricing)
- Chunk 3: Paragraphs 6-8 (all reviews)

Each chunk is topically coherent.

---

#### 23.2.5 Recursive Chunking

**ID:** `recursive-chunking`
**Parent:** `23.2`

**Full Explanation:**
Recursive chunking attempts to split documents using a hierarchy of separators, falling back to smaller separators only when needed. First tries to split on double newlines (paragraphs), then single newlines, then sentences, then words. This preserves document structure while ensuring chunks don't exceed size limits. Adapts to document format automatically.

**Simple Explanation:**
Try to split at natural breaks first. First try paragraphs. If still too big, try sentences. If still too big, try words. This keeps the most natural structure possible while meeting size limits.

**Example:**
Document: 2000 tokens

Recursive process:
1. Split by "\n\n" (paragraphs) → 5 chunks of ~400 tokens each ✓
   - Chunk 1: 350 tokens ✓
   - Chunk 2: 800 tokens ✗ (too big)
   
2. For chunk 2, split by "\n" (lines) → 3 sub-chunks
   - Sub-chunk 2a: 300 tokens ✓
   - Sub-chunk 2b: 250 tokens ✓
   - Sub-chunk 2c: 250 tokens ✓

Final: 7 chunks, all properly sized, natural boundaries preserved.

---

### 23.3 Embeddings

---

#### 23.3.1 Text Embeddings

**ID:** `text-embeddings`
**Parent:** `23.3`

**Full Explanation:**
Text embeddings are dense vector representations that capture semantic meaning of text. Similar meanings map to nearby vectors in embedding space, enabling semantic search beyond keyword matching. Embeddings are generated by neural networks trained on large corpora to understand language relationships. The embedding model's quality directly impacts retrieval accuracy.

**Simple Explanation:**
Convert text into numbers that capture meaning. "Dog" and "puppy" get similar numbers because they mean similar things. "Dog" and "refrigerator" get very different numbers. This lets us find documents by meaning, not just matching words.

**Example:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
texts = [
    "How to return a product",     # [0.12, -0.34, 0.56, ...]
    "Refund policy and procedures", # [0.11, -0.32, 0.58, ...] (similar!)
    "Best pizza recipes"           # [-0.45, 0.67, -0.23, ...] (different!)
]

embeddings = model.encode(texts)

# Similarity: texts[0] and texts[1] are close in vector space
# texts[2] is far from both
```

---

#### 23.3.2 Embedding Models

**ID:** `embedding-models`
**Parent:** `23.3`

**Full Explanation:**
Embedding models are neural networks trained to produce semantic text representations. Options range from open-source (Sentence Transformers, E5, BGE) to proprietary (OpenAI, Cohere, Voyage). Key factors: dimension size (256-3072), max input length, domain specialization, multilingual support, and cost. Model choice significantly impacts retrieval quality—larger models often perform better but cost more.

**Simple Explanation:**
Different "translators" that convert text to numbers. Some are free, some cost money. Some are better for general text, others for specific domains. Bigger models usually work better but are slower and more expensive.

**Example:**
Popular embedding models comparison:

| Model | Dimensions | Max Tokens | Best For |
|-------|------------|------------|----------|
| all-MiniLM-L6-v2 | 384 | 256 | Fast, general purpose |
| text-embedding-ada-002 | 1536 | 8191 | OpenAI, high quality |
| bge-large-en-v1.5 | 1024 | 512 | Open source, high quality |
| voyage-2 | 1024 | 4000 | Long documents |
| e5-large-v2 | 1024 | 512 | Strong retrieval |

```python
# OpenAI embeddings
from openai import OpenAI
client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input="Your text here"
)
embedding = response.data[0].embedding  # 1536-dim vector
```

---

#### 23.3.3 Cosine Similarity

**ID:** `cosine-similarity`
**Parent:** `23.3`

**Full Explanation:**
Cosine similarity measures the angle between two vectors, regardless of magnitude. Formula: cos(θ) = (A·B)/(||A||||B||). Range: -1 (opposite) to 1 (identical direction). Value of 0 means orthogonal (unrelated). Preferred for text embeddings because it's invariant to document length—a long document and short query can still match if they're about the same topic.

**Simple Explanation:**
Measure how similar two vectors are by their direction, ignoring length. If two arrows point the same way, similarity = 1. Opposite directions = -1. Perpendicular = 0. We use this because a short question and long answer can point the same "direction" (topic) even though they're different lengths.

**Example:**
```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Query: "refund policy"
query_embedding = [0.8, 0.6]

# Document 1: "How to get a refund"
doc1_embedding = [0.75, 0.65]

# Document 2: "Best pizza recipes"  
doc2_embedding = [-0.5, 0.7]

sim1 = cosine_similarity(query_embedding, doc1_embedding)  # 0.99 (very similar!)
sim2 = cosine_similarity(query_embedding, doc2_embedding)  # 0.14 (not similar)

# Retrieve doc1, not doc2
```

---

### 23.4 Vector Stores

---

#### 23.4.1 Vector Database

**ID:** `vector-database`
**Parent:** `23.4`

**Full Explanation:**
Vector databases are specialized systems optimized for storing, indexing, and querying high-dimensional vectors. Unlike traditional databases that search by exact match, vector databases find approximate nearest neighbors efficiently using specialized index structures (HNSW, IVF). Features include metadata filtering, hybrid search, real-time updates, and scalability to billions of vectors.

**Simple Explanation:**
A special database for storing embeddings (number lists). Regular databases find exact matches; vector databases find "similar" items even without exact matches. Like Google finding relevant results even if your search words don't exactly match any webpage.

**Example:**
```python
import chromadb

# Create client and collection
client = chromadb.Client()
collection = client.create_collection("my_documents")

# Add documents with embeddings
collection.add(
    documents=["Refund policy details...", "Shipping information..."],
    metadatas=[{"source": "policy.pdf"}, {"source": "shipping.pdf"}],
    ids=["doc1", "doc2"]
)

# Query by similarity
results = collection.query(
    query_texts=["How do I return an item?"],
    n_results=2
)
# Returns most similar documents
```

---

#### 23.4.2 HNSW (Hierarchical Navigable Small World)

**ID:** `hnsw`
**Parent:** `23.4`

**Full Explanation:**
HNSW is a graph-based index structure for approximate nearest neighbor search. It builds a hierarchical graph where each layer contains a subset of vectors, with the bottom layer containing all vectors. Search starts from top layer (few vectors, long-range connections), progressively moving to lower layers (more vectors, short-range connections). Provides excellent recall with logarithmic search complexity.

**Simple Explanation:**
A clever way to organize vectors for fast searching. Imagine a multi-story building: top floor has few landmarks for big jumps, lower floors have more detail for precise navigation. Start at the top, quickly get close to target, then refine on lower floors.

**Example:**
```
HNSW Index Structure:

Layer 2 (sparse):    [A]─────────────────[B]
                      │                    │
Layer 1 (medium):    [A]───[C]───[D]───[B]
                      │    │     │     │
Layer 0 (dense):    [A][E][C][F][D][G][B][H]

Search for query Q:
1. Layer 2: Start at A, jump to B (closer to Q)
2. Layer 1: From B, navigate B→D (closer to Q)
3. Layer 0: From D, find exact nearest neighbors [F, G]

Much faster than checking all vectors!
```

---

#### 23.4.3 Vector Database Options

**ID:** `vector-db-options`
**Parent:** `23.4`

**Full Explanation:**
Major vector database options include: Pinecone (managed, production-ready), Weaviate (open-source, hybrid search), Chroma (lightweight, embedded), Milvus (scalable, open-source), Qdrant (Rust-based, performant), pgvector (PostgreSQL extension), and FAISS (library, not database). Choice depends on scale, deployment preference (managed vs self-hosted), features needed, and existing infrastructure.

**Simple Explanation:**
Different tools for storing embeddings:
- **Pinecone**: Easy cloud service, just works, costs money
- **Chroma**: Simple, free, good for starting out
- **Weaviate**: Free, powerful, needs more setup
- **pgvector**: Use your existing PostgreSQL database
- **FAISS**: Facebook's library, very fast, not a full database

**Example:**
Choosing a vector database:

| Need | Recommendation |
|------|----------------|
| Quick prototype | Chroma (simple setup) |
| Production, no ops | Pinecone (managed) |
| Full control, open source | Weaviate or Milvus |
| Already using PostgreSQL | pgvector |
| Maximum speed, local | FAISS |
| Hybrid text + vector | Weaviate |

```python
# Chroma (simple)
import chromadb
client = chromadb.Client()

# Pinecone (managed)
import pinecone
pinecone.init(api_key="xxx")
index = pinecone.Index("my-index")

# pgvector (PostgreSQL)
# CREATE EXTENSION vector;
# CREATE TABLE items (embedding vector(1536));
```

---

### 23.5 Retrieval Methods

---

#### 23.5.1 Dense Retrieval

**ID:** `dense-retrieval`
**Parent:** `23.5`

**Full Explanation:**
Dense retrieval uses neural network embeddings to represent queries and documents as dense vectors, finding matches through vector similarity. It captures semantic meaning—synonyms and paraphrases can match even without shared words. Requires embedding model inference for queries and pre-computed document embeddings. Excels at semantic matching but may miss exact keyword matches.

**Simple Explanation:**
Find documents by meaning, not words. "Automobile" query finds "car" documents even though the words are different. Uses AI embeddings to understand what text means, then finds documents with similar meanings.

**Example:**
Query: "canine companions"

Dense retrieval matches:
- "Dogs make great pets" (semantically similar: dogs = canines)
- "Puppy training guide" (related concept)
- "Best dog food brands" (same topic)

Keyword search would find: Nothing (no word matches)

Dense retrieval understands "canine companions" ≈ "dogs"

---

#### 23.5.2 Sparse Retrieval

**ID:** `sparse-retrieval`
**Parent:** `23.5`

**Full Explanation:**
Sparse retrieval uses traditional keyword-based methods where documents are represented as high-dimensional sparse vectors (mostly zeros) with non-zero values for terms present. Algorithms like BM25 and TF-IDF weight term importance. Fast, interpretable, and excellent for exact matches and rare terms. Doesn't understand synonyms or semantics.

**Simple Explanation:**
Find documents by matching keywords. Fast and simple—look for documents containing the exact words in your query. Great for finding specific terms like product codes or names, but misses synonyms.

**Example:**
Query: "iPhone 15 Pro Max specs"

Sparse retrieval (BM25):
✓ Excellent at finding documents with exact phrase "iPhone 15 Pro Max"
✓ Good at rare/specific terms
✗ Won't find documents saying "Apple's latest smartphone specifications"

```python
from rank_bm25 import BM25Okapi

corpus = ["iPhone 15 Pro specs...", "Latest Apple phone...", "Android review"]
tokenized = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized)

query = "iPhone 15 Pro"
scores = bm25.get_scores(query.split())
# [0.8, 0.2, 0.0] - First doc matches best
```

---

#### 23.5.3 Hybrid Retrieval

**ID:** `hybrid-retrieval`
**Parent:** `23.5`

**Full Explanation:**
Hybrid retrieval combines dense (semantic) and sparse (keyword) retrieval, leveraging strengths of both. Methods include: linear combination of scores, reciprocal rank fusion, or re-ranking sparse results with dense model. Captures both semantic similarity and exact matches. Often outperforms either method alone, especially for diverse query types.

**Simple Explanation:**
Best of both worlds: use keyword matching AND meaning matching together. Some queries need exact words ("error code 404"), others need meaning ("how to fix website problem"). Hybrid handles both by combining scores from both methods.

**Example:**
Query: "Python pandas dataframe error 1234"

Dense retrieval alone:
- Finds general DataFrame tutorials (semantic match)
- Misses specific error code

Sparse retrieval alone:
- Finds error 1234 docs
- Misses relevant DataFrame debugging without exact words

Hybrid retrieval:
- Combines both sets
- Re-ranks to prioritize docs with BOTH semantic relevance AND keyword matches
- Best results!

```python
# Hybrid scoring
def hybrid_search(query, alpha=0.5):
    dense_results = dense_search(query)   # Semantic
    sparse_results = sparse_search(query)  # Keyword
    
    # Combine scores
    for doc in all_docs:
        doc.score = alpha * dense_results[doc] + (1-alpha) * sparse_results[doc]
    
    return sorted(all_docs, key=lambda x: x.score, reverse=True)
```

---

#### 23.5.4 Maximum Marginal Relevance (MMR)

**ID:** `mmr`
**Parent:** `23.5`

**Full Explanation:**
MMR balances relevance and diversity in retrieval results. It iteratively selects documents that are both relevant to the query AND different from already-selected documents. Formula: MMR = argmax[λ·Sim(d,q) - (1-λ)·max(Sim(d,d_selected))]. Prevents redundant results when top matches contain similar information. λ controls relevance-diversity tradeoff.

**Simple Explanation:**
Avoid repetition in results. If your top 5 results all say the same thing, that's wasteful. MMR picks diverse results: the most relevant one first, then the next most relevant that adds NEW information, and so on.

**Example:**
Query: "machine learning benefits"

Without MMR (top 3 by relevance only):
1. "ML improves efficiency and accuracy" 
2. "Machine learning increases efficiency" (redundant!)
3. "ML makes processes more efficient" (redundant!)

With MMR (λ=0.7):
1. "ML improves efficiency and accuracy" (most relevant)
2. "ML enables personalization at scale" (relevant + different)
3. "ML reduces costs through automation" (relevant + different)

Each result adds unique value.

---

### 23.6 Query Processing

---

#### 23.6.1 Query Expansion

**ID:** `query-expansion`
**Parent:** `23.6`

**Full Explanation:**
Query expansion augments the original query with additional related terms to improve retrieval recall. Techniques include: synonym addition, LLM-based expansion, pseudo-relevance feedback (extract terms from initial results), and knowledge graph expansion. Helps when user queries are brief or use different terminology than documents.

**Simple Explanation:**
Make the query bigger and better. User types "laptop"; expand to "laptop computer notebook PC portable". Now you'll find documents using any of these words. Especially helpful for short or vague queries.

**Example:**
Original query: "headache medicine"

Expanded query: "headache medicine pain relief aspirin ibuprofen acetaminophen migraine treatment"

```python
def expand_query(query, llm):
    prompt = f"""Generate 5 related search terms for: "{query}"
    Return as comma-separated list."""
    
    expansions = llm.generate(prompt)
    # "pain reliever, aspirin, ibuprofen, migraine remedy, analgesic"
    
    return f"{query} {expansions}"
```

Result: Finds documents about "pain relievers" and "aspirin" that the original query would miss.

---

#### 23.6.2 HyDE (Hypothetical Document Embeddings)

**ID:** `hyde`
**Parent:** `23.6`

**Full Explanation:**
HyDE generates a hypothetical answer to the query using an LLM, then uses that answer's embedding for retrieval instead of the query embedding. Hypothetical documents are more semantically similar to actual documents than short queries are. The generated answer doesn't need to be factually correct—it just needs to be in the same semantic space as relevant documents.

**Simple Explanation:**
Guess what a good answer might look like, then search for documents similar to that guess. A question like "What causes rain?" is short and vague. But if we first generate "Rain is caused by water evaporating and condensing..." that's much easier to match against actual documents.

**Example:**
Query: "Why is the sky blue?"

Standard retrieval:
- Embed short query → often matches poorly with long documents

HyDE approach:
1. Generate hypothetical answer:
   "The sky appears blue due to Rayleigh scattering, where 
    sunlight interacts with molecules in the atmosphere,
    scattering shorter blue wavelengths more than other colors."

2. Embed this hypothetical document (not the query)

3. Retrieve documents similar to hypothetical answer

Result: Better matches because we're comparing document-like text to documents.

---

#### 23.6.3 Query Decomposition

**ID:** `query-decomposition`
**Parent:** `23.6`

**Full Explanation:**
Query decomposition breaks complex questions into simpler sub-questions that can be answered independently, then synthesizes results. Useful for multi-hop questions requiring information from multiple documents. Methods include LLM-based decomposition and structured query parsing. Each sub-query retrieves focused context, and the final answer combines all pieces.

**Simple Explanation:**
Split hard questions into easier parts. "Compare iPhone and Samsung prices and features" becomes three questions: "iPhone price?", "Samsung price?", "Feature comparison?". Answer each, then combine into final response.

**Example:**
Complex query: "How did Apple's revenue compare to Microsoft's in 2023, and which company had better growth?"

Decomposed:
1. "What was Apple's revenue in 2023?"
2. "What was Microsoft's revenue in 2023?"
3. "What was Apple's revenue growth rate?"
4. "What was Microsoft's revenue growth rate?"

Each sub-query retrieves specific documents. Final synthesis:
"Apple's revenue was $383B (2% growth), Microsoft's was $211B (7% growth). While Apple had higher total revenue, Microsoft showed stronger growth."

---

### 23.7 Re-Ranking

---

#### 23.7.1 Re-Ranking Overview

**ID:** `reranking-overview`
**Parent:** `23.7`

**Full Explanation:**
Re-ranking is a two-stage retrieval approach: first retrieve many candidates quickly (e.g., top-100 via dense/sparse search), then use a more expensive model to re-score and reorder them. Re-rankers see the full query-document pair, enabling more accurate relevance scoring than embedding similarity alone. Improves precision among retrieved results.

**Simple Explanation:**
Get rough results first, then sort them more carefully. Stage 1: Quickly grab 100 possibly-relevant documents. Stage 2: Carefully read each one and put the best at the top. Like a quick skim followed by careful reading.

**Example:**
Query: "How to train a neural network"

Stage 1 - Fast retrieval (embedding similarity):
Returns 100 documents, roughly relevant

Stage 2 - Re-ranking (cross-encoder):
- Doc A: "Neural Network Training Guide" → Score: 0.95
- Doc B: "Deep Learning Basics" → Score: 0.82
- Doc C: "Brain Neuron Biology" → Score: 0.25 (not actually relevant!)

Re-ranked top 3 are much better than initial top 3.

---

#### 23.7.2 Cross-Encoder Re-Ranking

**ID:** `cross-encoder`
**Parent:** `23.7`

**Full Explanation:**
Cross-encoders process query and document together through a single transformer, enabling rich interaction between them. Unlike bi-encoders (separate embeddings), cross-encoders see both texts simultaneously and output a relevance score directly. Much more accurate but O(n) inference cost—must run for each query-document pair. Used for re-ranking, not initial retrieval.

**Simple Explanation:**
Read the question and document together to judge relevance. More accurate than comparing separate embeddings because it can see how specific parts of the question relate to specific parts of the document. Slow (can't pre-compute) but very accurate.

**Example:**
```python
from sentence_transformers import CrossEncoder

# Cross-encoder sees both together
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query = "What is the capital of France?"
documents = [
    "Paris is the capital and largest city of France.",
    "France is a country in Western Europe.",
    "The Eiffel Tower is in Paris."
]

# Score each pair
scores = model.predict([(query, doc) for doc in documents])
# [0.98, 0.23, 0.45]

# Document 1 wins (directly answers the question)
```

---

#### 23.7.3 Lost in the Middle Problem

**ID:** `lost-in-middle`
**Parent:** `23.7`

**Full Explanation:**
"Lost in the Middle" refers to LLMs' tendency to focus on information at the beginning and end of their context window, underweighting information in the middle. This affects RAG: even if relevant information is retrieved, placing it in the middle of the context may cause it to be ignored. Mitigation: put most relevant chunks at the start or end, or use retrieval strategies that optimize placement.

**Simple Explanation:**
LLMs pay most attention to the start and end of their context, ignoring the middle. If you put the most important document in the middle of 10 retrieved docs, the LLM might miss it. Solution: put the best stuff first or last.

**Example:**
Context with 5 retrieved chunks:
```
Chunk 1 (first): Weather information [HIGH ATTENTION]
Chunk 2: Irrelevant content
Chunk 3: THE ACTUAL ANSWER [LOW ATTENTION - Lost!]
Chunk 4: More irrelevant content  
Chunk 5 (last): Related but incomplete [HIGH ATTENTION]
```

Better ordering:
```
Chunk 3: THE ACTUAL ANSWER [HIGH ATTENTION]
Chunk 5: Related but incomplete
Chunk 1: Weather information
Chunk 2: Irrelevant content
Chunk 4: More irrelevant content
```

---

### 23.8 Context Management

---

#### 23.8.1 Context Window

**ID:** `context-window`
**Parent:** `23.8`

**Full Explanation:**
The context window is the maximum number of tokens an LLM can process in a single request, including system prompt, retrieved documents, user query, and generated response. Limits range from 4K (older models) to 128K+ (modern models like GPT-4 Turbo, Claude). Larger context allows more retrieved documents but increases cost and latency. Strategic context management optimizes information density within limits.

**Simple Explanation:**
How much text the AI can read at once. Like a desk that can only hold so many papers. Bigger context = more documents can be included, but costs more. You need to fit your question, relevant documents, and leave room for the answer.

**Example:**
GPT-4 Turbo with 128K context window:

Budget allocation:
- System prompt: 500 tokens
- Retrieved documents: 100,000 tokens (plenty!)
- User query: 100 tokens
- Response: 4,000 tokens
- Buffer: 23,400 tokens

GPT-3.5 with 4K context window:
- System prompt: 200 tokens
- Retrieved documents: 2,500 tokens (only ~2 pages!)
- User query: 100 tokens
- Response: 1,000 tokens
- Buffer: 200 tokens

Must choose retrieved content more carefully with smaller windows.

---

### 23.9 Generation

---

#### 23.9.1 Prompt Construction for RAG

**ID:** `rag-prompt-construction`
**Parent:** `23.9`

**Full Explanation:**
RAG prompts combine retrieved context with the user query in a structured format. Key elements: clear instructions on how to use context, retrieved documents with metadata, the user question, and output format guidance. Good prompts instruct the model to cite sources, acknowledge uncertainty, and stay grounded in provided context rather than using training knowledge.

**Simple Explanation:**
Build the message to the AI carefully. Include: "Here are some relevant documents, answer the question using only these, cite your sources, say 'I don't know' if the answer isn't in the documents."

**Example:**
```python
def construct_rag_prompt(query, retrieved_docs):
    context = "\n\n".join([
        f"[Document {i+1}]: {doc.content}" 
        for i, doc in enumerate(retrieved_docs)
    ])
    
    return f"""You are a helpful assistant. Answer questions based ONLY 
on the provided context. If the answer is not in the context, say 
"I don't have enough information to answer this question."

Cite your sources using [Document X] notation.

Context:
{context}

Question: {query}

Answer:"""
```

Output example:
"According to [Document 2], the refund policy allows returns within 30 days. [Document 1] clarifies that items must be in original packaging."

---

#### 23.9.2 Citation Generation

**ID:** `citation-generation`
**Parent:** `23.9`

**Full Explanation:**
Citation generation requires the LLM to reference source documents when making claims. Methods include: inline citations [1], footnotes, or explicit source attribution. This enables fact-checking, builds user trust, and makes hallucinations detectable. Advanced systems verify citations against sources post-generation and flag unsupported claims.

**Simple Explanation:**
Make the AI show its work. Every claim should say where it came from. "The policy allows 30-day returns [Source: policy.pdf]". Users can verify claims, and you can catch when the AI makes stuff up.

**Example:**
Without citations:
"The product costs $99 and includes free shipping."
(Is this true? Which document said this?)

With citations:
"The product costs $99 [Document 1: pricing.pdf] and includes free shipping for orders over $50 [Document 3: shipping.pdf]."

Verification:
- Claim 1: "$99" → Check pricing.pdf → ✓ Verified
- Claim 2: "free shipping" → Check shipping.pdf → ✓ Verified

---

### 23.10 Advanced RAG Patterns

---

#### 23.10.1 Naive RAG

**ID:** `naive-rag`
**Parent:** `23.10`

**Full Explanation:**
Naive RAG is the basic retrieval-augmented generation pattern: embed query → retrieve top-K chunks → concatenate with query → generate response. Simple to implement but has limitations: no query understanding, fixed retrieval strategy, no result refinement, and sensitivity to retrieval quality. Serves as baseline for more advanced approaches.

**Simple Explanation:**
The simplest RAG: search for documents, stuff them in the prompt, ask the LLM. Quick to build but not very smart. Doesn't improve the query, doesn't check if results are good, doesn't verify the answer.

**Example:**
```python
# Naive RAG
def naive_rag(query, vectorstore, llm):
    # 1. Retrieve (simple similarity search)
    docs = vectorstore.similarity_search(query, k=4)
    
    # 2. Construct prompt (basic concatenation)
    context = "\n".join([doc.content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    # 3. Generate (no verification)
    return llm.generate(prompt)
```

Limitations:
- No query improvement
- Fixed k=4 retrieval
- No re-ranking
- No answer verification

---

#### 23.10.2 Advanced RAG

**ID:** `advanced-rag`
**Parent:** `23.10`

**Full Explanation:**
Advanced RAG adds pre-retrieval and post-retrieval optimizations: query rewriting/expansion before retrieval, hybrid search combining dense and sparse methods, re-ranking retrieved results, context compression to remove irrelevant parts, and answer refinement/verification. Each stage can be independently optimized. Significantly improves quality over naive RAG.

**Simple Explanation:**
Smarter RAG with extra steps: improve the question first, use better search, sort results carefully, compress context to essentials, check the answer makes sense. More complex but much better results.

**Example:**
```python
# Advanced RAG Pipeline
def advanced_rag(query, vectorstore, llm, reranker):
    # PRE-RETRIEVAL
    # 1. Query expansion
    expanded_query = llm.expand_query(query)
    
    # 2. Hybrid search (dense + sparse)
    dense_results = vectorstore.similarity_search(expanded_query, k=20)
    sparse_results = bm25_search(expanded_query, k=20)
    candidates = merge_results(dense_results, sparse_results)
    
    # POST-RETRIEVAL  
    # 3. Re-rank
    reranked = reranker.rerank(query, candidates)[:5]
    
    # 4. Compress context
    compressed = compress_context(query, reranked)
    
    # 5. Generate with verification
    answer = llm.generate_with_citations(query, compressed)
    
    # 6. Verify citations
    verified_answer = verify_citations(answer, reranked)
    
    return verified_answer
```

---

#### 23.10.3 Self-RAG

**ID:** `self-rag`
**Parent:** `23.10`

**Full Explanation:**
Self-RAG trains the LLM to decide when retrieval is needed and to critique its own outputs. The model generates special tokens indicating: whether retrieval is necessary, whether retrieved docs are relevant, whether the response is supported by evidence, and response utility. This enables adaptive retrieval (only when needed) and self-correction without external verification.

**Simple Explanation:**
The AI decides when it needs to look things up and judges its own answers. Instead of always retrieving, it first asks "do I need external info?" Then it checks "is this document relevant?" and "is my answer supported?" Self-aware RAG.

**Example:**
```
Query: "What is 2+2?"
Self-RAG: [No Retrieval Needed] → Generates "4" directly

Query: "What was Apple's Q4 2023 revenue?"
Self-RAG: [Retrieval Needed] → Retrieves documents
         [Document Relevant: Yes] → Uses document
         [Response Supported: Yes] → "$89.5 billion [Source]"
         [Utility: High] → Returns response

Query: "What is the meaning of life?"
Self-RAG: [Retrieval Needed] → Retrieves documents
         [Document Relevant: No] → Discards retrieved docs
         [Response Supported: Partially] → Flags uncertainty
```

---

#### 23.10.4 Corrective RAG (CRAG)

**ID:** `crag`
**Parent:** `23.10`

**Full Explanation:**
Corrective RAG (CRAG) evaluates retrieval quality and takes corrective actions when retrieval fails. It grades retrieved documents as Correct, Ambiguous, or Incorrect. For incorrect retrievals, CRAG triggers web search or other knowledge sources as fallback. For ambiguous cases, it combines retrieved and web-searched information. Improves robustness when initial retrieval quality is poor.

**Simple Explanation:**
RAG that checks if retrieval worked and fixes it if not. After retrieving documents, it asks "are these actually relevant?" If not, it tries a different source like web search. Self-correcting when the first attempt fails.

**Example:**
```python
def corrective_rag(query, vectorstore, llm, web_search):
    # Initial retrieval
    docs = vectorstore.similarity_search(query, k=5)
    
    # Grade retrieval quality
    grade = llm.grade_relevance(query, docs)
    
    if grade == "CORRECT":
        # Good retrieval, proceed normally
        return generate_answer(query, docs)
    
    elif grade == "INCORRECT":
        # Retrieval failed, use web search
        web_docs = web_search(query)
        return generate_answer(query, web_docs)
    
    elif grade == "AMBIGUOUS":
        # Uncertain, combine both sources
        web_docs = web_search(query)
        combined = docs + web_docs
        return generate_answer(query, combined)
```

---

#### 23.10.5 Agentic RAG

**ID:** `agentic-rag`
**Parent:** `23.10`

**Full Explanation:**
Agentic RAG uses autonomous agents that can plan multi-step retrieval strategies, use tools, and iteratively refine results. Agents decide: which sources to query, how to reformulate queries based on initial results, when to stop retrieving, and how to synthesize information from multiple retrieval rounds. Enables complex research tasks requiring reasoning about information needs.

**Simple Explanation:**
RAG with an AI agent that plans and executes its own research strategy. Instead of one search, the agent might: search, read results, decide it needs more info, search differently, find contradictions, search again to resolve them, then write final answer. Like a research assistant, not just a search engine.

**Example:**
```python
# Agentic RAG with planning
class RAGAgent:
    def answer(self, query):
        plan = self.plan_research(query)
        # Plan: ["Search company docs", "Find financial reports", 
        #        "Cross-reference with news"]
        
        knowledge = {}
        for step in plan:
            results = self.execute_step(step)
            knowledge = self.integrate(knowledge, results)
            
            # Agent decides if more research needed
            if self.is_sufficient(query, knowledge):
                break
            else:
                # Adapt plan based on findings
                plan = self.replan(query, knowledge, plan)
        
        return self.synthesize_answer(query, knowledge)
```

Query: "Why did Company X's stock drop last week?"

Agent execution:
1. Search internal docs → Found Q4 earnings report
2. Agent realizes: "Need external news context"
3. Search news → Found negative analyst reports
4. Agent notices: "Conflicting information, need more sources"
5. Search SEC filings → Found insider selling disclosure
6. Agent: "Have enough context now"
7. Synthesize: "Stock dropped due to weak Q4 earnings, negative analyst coverage, and insider selling..."

---

#### 23.10.6 Graph RAG

**ID:** `graph-rag`
**Parent:** `23.10`

**Full Explanation:**
Graph RAG enhances retrieval using knowledge graph structures. Documents are processed to extract entities and relationships, forming a graph. Retrieval traverses the graph to find connected information that vector similarity might miss. Enables multi-hop reasoning—finding information through chains of relationships. Particularly useful for complex queries requiring connected facts.

**Simple Explanation:**
RAG using a web of connected facts instead of just similar documents. Build a map of "Company X → employs → Person Y → invented → Product Z". When asked about Product Z, traverse the graph to find all connected relevant information, even if the documents don't mention Z directly.

**Example:**
Query: "What products has the Stanford AI Lab influenced?"

Traditional RAG: Search for "Stanford AI Lab products" → Might miss connections

Graph RAG:
```
Knowledge Graph:
[Stanford AI Lab]──created──[ImageNet]
        │                        │
    employs                  influenced
        │                        │
    [Fei-Fei Li]           [AlexNet]
        │                        │
     advised               enabled
        │                        │
    [Andrej Karpathy]      [CNN Revolution]
        │
     created
        │
    [Tesla Autopilot]
```

Graph traversal finds:
- ImageNet (direct connection)
- AlexNet (influenced by ImageNet)
- Tesla Autopilot (via Karpathy connection)

Retrieves documents for each discovered entity.

---

### 23.11 RAG Evaluation

---

#### 23.11.1 RAG Evaluation Overview

**ID:** `rag-evaluation-overview`
**Parent:** `23.11`

**Full Explanation:**
RAG evaluation assesses both retrieval and generation components. Retrieval metrics: precision, recall, MRR for finding relevant documents. Generation metrics: faithfulness (grounded in context), relevance (answers the question), correctness (factually accurate). End-to-end metrics evaluate the complete pipeline. Evaluation can be automated (LLM-as-judge) or human-based.

**Simple Explanation:**
Measure if RAG is working well. Two parts: (1) Is retrieval finding the right documents? (2) Is generation using them correctly? Need metrics for both, plus overall "did we answer the question correctly?"

**Example:**
RAG Evaluation Framework:

| Component | Metric | What it Measures |
|-----------|--------|------------------|
| Retrieval | Recall@5 | Found relevant docs in top 5? |
| Retrieval | MRR | How high is first relevant doc? |
| Generation | Faithfulness | Answer supported by context? |
| Generation | Relevance | Answer addresses question? |
| End-to-End | Correctness | Final answer is right? |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall]
)
# {'faithfulness': 0.89, 'answer_relevancy': 0.92, 'context_recall': 0.85}
```

---

#### 23.11.2 Faithfulness

**ID:** `faithfulness-metric`
**Parent:** `23.11`

**Full Explanation:**
Faithfulness measures whether the generated response is factually consistent with the retrieved context—no hallucinated information beyond what's in the documents. Evaluation: decompose response into claims, verify each claim against context. Score is the proportion of claims supported by context. Critical metric for preventing hallucinations.

**Simple Explanation:**
Does the answer stick to what the documents actually say? If the document says "revenue was $100M" but the answer says "$150M," that's unfaithful. Every claim in the answer should be traceable to the retrieved context.

**Example:**
Retrieved context:
"Company X had revenue of $100M in 2023. They employ 500 people."

Generated answer:
"Company X generated $100M revenue in 2023 with 500 employees. 
 They are the market leader."

Faithfulness evaluation:
- Claim 1: "$100M revenue" → In context ✓
- Claim 2: "500 employees" → In context ✓  
- Claim 3: "market leader" → NOT in context ✗

Faithfulness score: 2/3 = 0.67

---

#### 23.11.3 Answer Relevance

**ID:** `answer-relevance-metric`
**Parent:** `23.11`

**Full Explanation:**
Answer relevance measures how well the generated response addresses the user's question, regardless of factual accuracy. Evaluation: generate potential questions that the answer addresses, compare semantic similarity with original question. High relevance means the answer is on-topic and complete. Low relevance indicates tangential or incomplete responses.

**Simple Explanation:**
Does the answer actually answer the question? Even if the answer is factually correct and grounded, if it doesn't address what was asked, it's not relevant. "What time is it?" answered with "The sky is blue" has zero relevance.

**Example:**
Question: "What is the company's refund policy?"

Answer A: "Returns are accepted within 30 days for full refund. Items must be unopened."
Relevance: HIGH (directly answers the question)

Answer B: "The company was founded in 1990 in California."
Relevance: LOW (doesn't address refund policy at all)

Answer C: "For product inquiries, contact support@company.com"
Relevance: MEDIUM (related but doesn't answer the specific question)

---

#### 23.11.4 Context Relevance

**ID:** `context-relevance-metric`
**Parent:** `23.11`

**Full Explanation:**
Context relevance evaluates whether retrieved documents are actually relevant to answering the query. Measures retrieval quality independent of generation. Irrelevant context wastes context window space and can confuse the generator. Evaluation: proportion of retrieved chunks that contain information useful for answering the question.

**Simple Explanation:**
Are the retrieved documents actually useful? If you asked about refund policies but retrieval returned documents about company history, context relevance is low. Good retrieval = high context relevance.

**Example:**
Query: "How to reset password?"

Retrieved documents:
1. "To reset your password, go to Settings > Security > Reset" ✓ Relevant
2. "Password must be 8+ characters with numbers" ✓ Relevant
3. "Company was founded in 2010" ✗ Not relevant
4. "Our office is in San Francisco" ✗ Not relevant

Context Relevance: 2/4 = 0.50 (poor retrieval quality)

---

#### 23.11.5 RAGAS Framework

**ID:** `ragas`
**Parent:** `23.11`

**Full Explanation:**
RAGAS (Retrieval-Augmented Generation Assessment) is an open-source framework for evaluating RAG pipelines. It provides metrics for context relevance, faithfulness, answer relevance, and context recall. Uses LLM-as-judge for scalable automated evaluation. Enables systematic comparison of RAG configurations and continuous monitoring in production.

**Simple Explanation:**
A toolkit for measuring RAG quality automatically. Instead of manually checking every answer, RAGAS uses AI to evaluate faithfulness, relevance, and correctness at scale. Industry standard for RAG evaluation.

**Example:**
```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset

# Prepare evaluation data
eval_data = {
    "question": ["What is the refund policy?"],
    "answer": ["Refunds are available within 30 days..."],
    "contexts": [["Document about refund policy..."]],
    "ground_truth": ["30-day refund for unopened items"]
}
dataset = Dataset.from_dict(eval_data)

# Evaluate
results = evaluate(
    dataset,
    metrics=[context_precision, context_recall, 
             faithfulness, answer_relevancy]
)

print(results)
# {'context_precision': 0.92, 'context_recall': 0.88,
#  'faithfulness': 0.95, 'answer_relevancy': 0.90}
```

---

### 23.12 RAG Infrastructure

---

#### 23.12.1 LangChain

**ID:** `langchain`
**Parent:** `23.12`

**Full Explanation:**
LangChain is a framework for building LLM-powered applications, including RAG systems. It provides abstractions for document loaders, text splitters, embedding models, vector stores, retrievers, and chains. Enables composable pipelines connecting these components. Extensive integrations with 100+ tools, models, and databases. Most popular framework for RAG development.

**Simple Explanation:**
A toolbox for building AI applications. Instead of writing everything from scratch, LangChain gives you ready-made pieces: document loaders, embedding generators, vector database connectors. Snap them together like Lego blocks to build RAG systems quickly.

**Example:**
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# 1. Load documents
loader = PyPDFLoader("company_docs.pdf")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Create RAG chain
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 5. Query
result = qa_chain.invoke({"query": "What is the refund policy?"})
print(result["result"])
```

---

#### 23.12.2 LlamaIndex

**ID:** `llamaindex`
**Parent:** `23.12`

**Full Explanation:**
LlamaIndex (formerly GPT Index) is a data framework for building RAG applications with emphasis on data indexing and retrieval strategies. Provides specialized index types (vector, keyword, tree, knowledge graph), advanced query engines, and tools for structured data. Focuses on sophisticated retrieval patterns and multi-document reasoning.

**Simple Explanation:**
Another framework for building RAG, focused on smart ways to organize and search your data. Specializes in advanced retrieval strategies like combining multiple indexes, querying across different document types, and building knowledge graphs.

**Example:**
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

# Configure
Settings.llm = OpenAI(model="gpt-4")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# 1. Load documents
documents = SimpleDirectoryReader("./data").load_data()

# 2. Create index
index = VectorStoreIndex.from_documents(documents)

# 3. Create query engine
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize"  # Advanced response synthesis
)

# 4. Query
response = query_engine.query("What is the refund policy?")
print(response)
print(response.source_nodes)  # Show sources
```

---

#### 23.12.3 RAG in Production

**ID:** `rag-production`
**Parent:** `23.12`

**Full Explanation:**
Production RAG requires considerations beyond prototype: latency optimization (async operations, caching, model selection), cost management (embedding API costs, LLM token costs), reliability (error handling, fallbacks, monitoring), scalability (vector database scaling, load balancing), and observability (logging queries, evaluating responses, detecting drift).

**Simple Explanation:**
Making RAG work in the real world. Prototypes are easy; production is hard. Need to think about: speed (users won't wait), cost (API bills add up), reliability (can't crash), scale (handle many users), and monitoring (know when things break).

**Example:**
Production RAG Checklist:

| Aspect | Considerations |
|--------|----------------|
| **Latency** | Cache embeddings, use faster models, async retrieval |
| **Cost** | Batch embeddings, smaller models where possible, cache responses |
| **Reliability** | Fallback to web search, graceful degradation, retry logic |
| **Scale** | Distributed vector DB, horizontal scaling, queue heavy requests |
| **Monitoring** | Log all queries, track retrieval quality, alert on failures |
| **Security** | Sanitize inputs, filter outputs, access control |

```python
# Production RAG with caching and monitoring
class ProductionRAG:
    def __init__(self):
        self.cache = RedisCache()
        self.metrics = PrometheusMetrics()
        self.fallback = WebSearch()
    
    async def query(self, question):
        # Check cache
        cached = await self.cache.get(question)
        if cached:
            self.metrics.record("cache_hit")
            return cached
        
        try:
            with self.metrics.timer("rag_latency"):
                result = await self.rag_pipeline(question)
            
            await self.cache.set(question, result, ttl=3600)
            self.metrics.record("success")
            return result
            
        except Exception as e:
            self.metrics.record("error")
            # Fallback to web search
            return await self.fallback.search(question)
```

---

## SUMMARY

This RAG section covers:

| Subsection | Topics |
|------------|--------|
| 23.1 Fundamentals | RAG overview, vs fine-tuning, architecture, pipeline, grounding, hallucination |
| 23.2 Document Processing | Chunking strategies, chunk size/overlap, semantic chunking |
| 23.3 Embeddings | Text embeddings, models, similarity metrics |
| 23.4 Vector Stores | Vector databases, HNSW, popular options |
| 23.5 Retrieval Methods | Dense, sparse, hybrid, MMR |
| 23.6 Query Processing | Expansion, HyDE, decomposition |
| 23.7 Re-Ranking | Cross-encoders, lost in middle |
| 23.8 Context Management | Context windows, token limits |
| 23.9 Generation | Prompt construction, citations |
| 23.10 Advanced Patterns | Naive/Advanced/Self/Corrective/Agentic/Graph RAG |
| 23.11 Evaluation | Faithfulness, relevance, RAGAS |
| 23.12 Infrastructure | LangChain, LlamaIndex, production |

**Total RAG concepts: ~80+ items with full explanations**

This section should be integrated into the main ML Cheatsheet Content Specification as Section 23.
