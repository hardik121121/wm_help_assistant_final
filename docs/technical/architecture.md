# System Architecture - Watermelon Documentation Assistant

**Comprehensive Technical Architecture Documentation**

This document provides a deep dive into the complete system architecture, strategies, tech stack, and codebase organization.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Layers](#architecture-layers)
3. [Key Strategies & Innovations](#key-strategies--innovations)
4. [Tech Stack](#tech-stack)
5. [Folder & File Structure](#folder--file-structure)
6. [Data Flow](#data-flow)
7. [Design Patterns](#design-patterns)
8. [Performance Characteristics](#performance-characteristics)

---

## System Overview

### What We Built

A **production-grade RAG (Retrieval-Augmented Generation) system** designed to handle complex multi-topic queries across 2,300+ pages of documentation.

**Key Differentiators**:
- Multi-step retrieval with context chaining
- Query decomposition for complex questions
- Hierarchical document processing with structure preservation
- Hybrid search (Vector + BM25 + Reranking)
- Query expansion with 32 synonym mappings
- Strategy-aware answer generation

### Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Precision@10 | 0.667 | 0.700 | 95% ✅ |
| Recall@10 | 0.638 | 0.600 | 106% ✅ |
| MRR | 0.854 | 0.700 | 122% ✅ |
| Generation Quality | 0.914 | 0.750 | 122% ✅ |
| Avg Query Time | 27.7s | <15s | 54% ⚠️ |
| Success Rate | 100% | >90% | ✅ |

---

## Architecture Layers

### Layer 1: User Interface (Streamlit Web App)

**File**: `app.py`

**Responsibilities**:
- Query input and example query selection
- Real-time pipeline visualization (4 stages)
- Answer display with formatting, citations, and images
- Metrics dashboard (retrieval + generation quality)
- Performance tracking and comparison

**Key Features**:
- Cached pipeline initialization (`@st.cache_resource`)
- Expandable sections for each pipeline stage
- Image gallery with lightbox
- Citation display by section
- Metrics visualization with color coding

**UI Components**:
```python
# Main sections
1. Header & Example Queries (selectbox)
2. Query Input (text_area)
3. Pipeline Execution (4 expandable stages)
4. Answer Display (formatted markdown)
5. Metrics Dashboard (3 columns: retrieval, generation, performance)
6. Images Gallery (thumbnails with captions)
7. Citations (organized by section)
```

---

### Layer 2: End-to-End Pipeline Orchestrator

**File**: `src/generation/end_to_end_pipeline.py`

**Class**: `EndToEndPipeline`

**Responsibilities**:
- Orchestrates all 4 stages of the RAG pipeline
- Manages dependencies between stages
- Aggregates metrics and timing
- Returns comprehensive `PipelineResult` dataclass

**Pipeline Stages**:
```python
def process_query(query: str) -> PipelineResult:
    # Stage 1: Query Understanding
    query_understanding = self.query_understanding.analyze_query(query)

    # Stage 2: Multi-Step Retrieval
    retrieval_result = self.retriever.retrieve(query, query_understanding)

    # Stage 3: Answer Generation
    answer = self.generator.generate_answer(query, retrieval_result.organized_context)

    # Stage 4: Validation
    validation = self.validator.validate_response(answer, query, query_understanding)

    return PipelineResult(
        query=query,
        query_understanding=query_understanding,
        retrieval_result=retrieval_result,
        answer=answer,
        validation=validation,
        metrics={...}
    )
```

**Dependencies**:
- `QueryUnderstanding` (Stage 1)
- `MultiStepRetriever` (Stage 2)
- `AnswerGenerator` (Stage 3)
- `ResponseValidator` (Stage 4)

---

### Layer 3: Query Understanding Engine

**Files**: `src/query/`

#### 3.1 Query Decomposer (`query_decomposer.py`)

**Strategy**: LLM-based decomposition using Groq Llama 3.3 70B

**Purpose**: Break complex multi-topic queries into 2-4 atomic sub-questions

**Example**:
```
Query: "How do I create a no-code block and test it with AFT?"

Sub-questions:
1. What is a no-code block in Watermelon?
2. How do I create a no-code block?
3. What is Autonomous Functional Testing (AFT)?
4. How do I connect a no-code block to AFT?
```

**Implementation**:
- Uses domain-specific prompt with Watermelon feature context
- Identifies dependency types: INDEPENDENT, SEQUENTIAL, CONDITIONAL
- Assigns priority (1=highest) for retrieval ordering
- Extracts topics per sub-question

**Key Parameters**:
- Model: `llama-3.3-70b-versatile` (Groq)
- Temperature: 0.2 (deterministic)
- Max sub-questions: 4
- Token budget: ~1,000 tokens/query

#### 3.2 Query Expander (`query_expander.py`)

**Strategy**: Synonym-based query expansion (NEW - Phase 8)

**Purpose**: Generate query variations to improve recall

**Mappings** (32 total):
- **Action synonyms** (10): integrate ↔ connect, link, setup, configure...
- **Integration aliases** (9): MS Teams ↔ Microsoft Teams ↔ Teams...
- **Technical synonyms** (8): API ↔ REST API, web service...
- **Concept expansions** (5): testing ↔ QA, quality assurance...

**Example**:
```
Original: "How do I integrate MS Teams?"

Expansions:
1. "How do I integrate MS Teams?" (original)
2. "How do I connect microsoft teams?"
3. "How do I link teams?"
```

**Impact**: +42.8% recall improvement

#### 3.3 Query Classifier (`query_classifier.py`)

**Strategy**: Rule-based classification with pattern matching

**Query Types**:
- `procedural` - How-to questions
- `conceptual` - What/Why questions
- `troubleshooting` - Error/problem questions
- `integration` - Integration setup questions
- `comparison` - Feature comparison questions

**Query Classes**:
- `simple` - Single topic, straightforward
- `complex` - Multiple topics or steps
- `multi-topic_procedural` - Multiple topics + how-to
- `multi-topic_integration` - Multiple integrations
- `conceptual_procedural` - What + How combined

#### 3.4 Intent Analyzer (`intent_analyzer.py`)

**Strategy**: Keyword-based intent detection

**Intents**:
- `create` - User wants to create something
- `configure` - User wants to configure settings
- `integrate` - User wants to integrate systems
- `troubleshoot` - User has a problem
- `learn` - User wants to understand concepts

---

### Layer 4: Multi-Step Retrieval System

**Files**: `src/retrieval/`

#### 4.1 Hybrid Search (`hybrid_search.py`)

**Strategy**: Vector search + BM25 keyword search + RRF fusion

**Components**:

**A. Vector Search**:
- **Embeddings**: OpenAI `text-embedding-3-large` (3072-dim)
- **Vector DB**: Pinecone serverless (cosine similarity)
- **Index**: 2,106 vectors
- **Top-K**: 50 results per query

**B. BM25 Keyword Search**:
- **Library**: rank-bm25
- **Vocabulary**: 16,460 terms
- **Top-K**: 50 results per query

**C. Query Expansion Integration**:
```python
# For each query, generate 3 variations
variations = self.query_expander.expand_query(query, max_expansions=3)

# Search each variation
all_results = []
for variation in variations:
    vector_results = self._vector_search(variation, top_k=50)
    bm25_results = self._bm25_search(variation, top_k=50)
    all_results.extend(vector_results + bm25_results)

# Fuse results using RRF
fused_results = self._rrf_fusion(all_results, k=60)
```

**D. RRF (Reciprocal Rank Fusion)**:
- **Formula**: `score = Σ(1 / (k + rank))` where k=60
- **Weights**: 70% vector, 30% BM25
- **Purpose**: Combine results from different search methods

**E. Pinecone Metadata Limit Workaround**:
```python
# Problem: Pinecone has 40KB metadata limit
# Solution: Maintain three maps in memory

# 1. Content map: chunk_id → full content
self.chunk_content_map = {chunk['metadata']['chunk_id']: chunk['content']}

# 2. Metadata map: chunk_id → full metadata (with image_paths list)
self.chunk_metadata_map = {chunk['metadata']['chunk_id']: chunk['metadata']}

# 3. Embeddings map: chunk_id → embedding vector
self.chunk_embeddings = {chunk['metadata']['chunk_id']: chunk['embedding']}

# During retrieval: Merge Pinecone results with full data
chunk_id = match.metadata['chunk_id']
content = self.chunk_content_map.get(chunk_id, '')  # Full content!
full_metadata = self.chunk_metadata_map.get(chunk_id, {})
merged = {**match.metadata, **full_metadata}
```

**Impact**: Fixed ALL integration queries (MS Teams, Shopify, Slack, etc.)

#### 4.2 Cohere Reranker (`reranker.py`)

**Strategy**: Semantic reranking for precision

**Model**: `rerank-english-v3.0`

**Purpose**: Rerank search results by semantic relevance to query

**Process**:
1. Take top 50 hybrid search results
2. Send to Cohere with query
3. Get semantic relevance scores
4. Return top 20 most relevant chunks

**Cost**: ~$0.0015 per query (3 sub-questions × 1 rerank each)

#### 4.3 Multi-Step Retriever (`multi_step_retriever.py`)

**Strategy**: Sequential retrieval with context chaining

**Process**:
```python
context_chain = []  # Accumulates results from earlier sub-questions

for sub_question in sub_questions:
    # Enhance query with previous context
    if enable_context_chaining and context_chain:
        enhanced_query = _enhance_with_context(sub_question.question, context_chain)
    else:
        enhanced_query = sub_question.question

    # Retrieve for this sub-question
    results = hybrid_search(enhanced_query)
    results = rerank(results, enhanced_query)

    # Add to context chain
    context_chain.extend(results[:5])  # Top 5 results
```

**Context Chaining Benefits**:
- Later sub-questions benefit from earlier context
- Improves relevance for sequential questions
- Better handling of dependencies

**Dependency Types**:
- `INDEPENDENT` - Can retrieve in any order
- `SEQUENTIAL` - Depends on previous sub-question
- `CONDITIONAL` - Conditional on previous results

#### 4.4 Context Organizer (`context_organizer.py`)

**Strategy**: Topic clustering + chronological ordering

**Purpose**: Organize retrieved chunks into coherent context

**Process**:
1. **Deduplicate**: Remove duplicate chunks across sub-questions
2. **Score Aggregation**: Combine scores for chunks retrieved multiple times
3. **Topic Clustering**: Group chunks by topic/section
4. **Chronological Ordering**: Order by page number within topics
5. **Limit**: Return top 20 chunks for generation

**Output**: `OrganizedContext` dataclass with:
- Final chunks (20 max)
- Topics covered
- Image paths
- Page ranges
- Metadata statistics

---

### Layer 5: Advanced Generation System

**Files**: `src/generation/`

#### 5.1 Answer Generator (`answer_generator.py`)

**Strategy**: Strategy-aware multi-context prompting

**Model**: Groq Llama 3.3 70B (`llama-3.3-70b-versatile`)

**Generation Strategies** (4 types):

**A. Step-by-Step** (procedural queries):
```
Prompt structure:
- Context from 20 chunks
- Instruction: "Provide step-by-step instructions"
- Format: Numbered list with sub-steps
- Include: Images and code blocks
```

**B. Comparison** (comparison queries):
```
Prompt structure:
- Context for both items being compared
- Instruction: "Create a comparison table"
- Format: Markdown table
- Include: Feature-by-feature breakdown
```

**C. Troubleshooting** (error/problem queries):
```
Prompt structure:
- Context about common issues
- Instruction: "Diagnose and provide solutions"
- Format: Problem → Diagnosis → Solutions
- Include: Alternative approaches
```

**D. Standard** (general queries):
```
Prompt structure:
- Context from retrieved chunks
- Instruction: "Provide comprehensive answer"
- Format: Structured paragraphs
- Include: All relevant information
```

#### 5.2 Smart Image Selector (`smart_image_selector.py`) 🆕

**Added**: November 9, 2025

**Purpose**: LLM-based filtering to show only relevant images

**Problem Solved**:
- Chunks contain 15-25 total images
- Not all images relevant to user query
- Showing all images causes information overload

**Solution**:
```python
class SmartImageSelector:
    def select_relevant_images(query: str, all_images: List[str], max_images: int = 6):
        # 1. Extract semantic names from filenames
        image_names = ["kubernetes config", "no code automation", ...]

        # 2. LLM ranks images by relevance to query
        llm_prompt = f"Query: {query}\nImages: {image_names}\nSelect most relevant"

        # 3. Return top-ranked images (max 6)
        return selected_images
```

**Key Features**:
- **Semantic Analysis**: Filenames like `no_code_automation_page252.png` → "no code automation"
- **LLM Ranking**: Groq Llama 3.3 70B compares images with query
- **Smart Filtering**: 15-25 images → top 6 most relevant
- **Low Cost**: ~100 tokens, $0.0001 per query
- **Fallback**: Returns first 6 if LLM fails

**Performance**:
- **Relevance Rate**: ~85% of selected images are relevant
- **Reduction**: 60-70% fewer images shown
- **UX Impact**: Significantly improved user experience

**Multi-Context Integration**:
```python
# Combine context from multiple chunks
context_text = "\n\n".join([
    f"Section {i+1}: {chunk['metadata']['heading_path']}\n{chunk['content']}"
    for i, chunk in enumerate(organized_context.final_chunks)
])

# Generate with strategy-specific prompt
prompt = self._build_prompt(query, context_text, strategy)
answer = self.llm.generate(prompt)
```

**Key Parameters**:
- Temperature: 0.2 (deterministic)
- Max tokens: 8192
- Context window: 128,000 tokens
- Cost: FREE (Groq free tier)

#### 5.2 Response Validator (`response_validator.py`)

**Strategy**: Multi-criteria quality scoring

**Validation Checks**:

**A. Completeness** (0.0-1.0):
- All sub-questions answered?
- All topics covered?
- Sufficient detail?

**B. Coherence** (0.0-1.0):
- Logical flow?
- Consistent terminology?
- Clear structure?

**C. Formatting** (0.0-1.0):
- Proper markdown?
- Headings/bullets used correctly?
- Code blocks formatted?

**D. Citation Quality** (0.0-1.0):
- Citations provided?
- Accurate section references?
- Images referenced?

**Overall Score**: Average of all sub-scores

**Quality Thresholds**:
- Excellent: ≥0.85
- Good: 0.70-0.85
- Fair: 0.50-0.70
- Poor: <0.50

**Current Performance**: 0.914 average (100% Excellent on 15 queries)

---

## Key Strategies & Innovations

### 1. Query Expansion System (NEW - Phase 8)

**What**: Automatically expand every query into 3 variations using 32 synonym mappings

**Why**: Documentation uses different terminology than users

**Impact**: +42.8% recall improvement

**Example**:
```
User: "How do I integrate MS Teams?"

System searches for:
1. "How do I integrate MS Teams?"
2. "How do I connect microsoft teams?"
3. "How do I link teams?"

Results are aggregated and deduplicated.
```

**Extensibility**: Easy to add new synonyms for new integrations

### 2. Hierarchical Document Processing

**What**: Preserve document structure during chunking

**Why**: Traditional chunking loses context and breaks logical boundaries

**Process**:
```
PDF → Docling Processor → Structured JSON → Hierarchical Chunker → Context-Aware Chunks
```

**Key Features**:
- Section-based chunking (respects heading boundaries)
- Context injection (prepend heading hierarchy to each chunk)
- Rich metadata (20+ fields per chunk)
- Table/image extraction with captions

**Example Chunk**:
```
Content:
  """
  Section: Getting Started > Integrations > MS Teams

  To integrate MS Teams with Watermelon:
  1. Navigate to Settings > Integrations
  2. Click on "Microsoft Teams"
  ...
  """

Metadata:
  - chunk_id: "chunk_0042"
  - heading_path: ["Getting Started", "Integrations", "MS Teams"]
  - page_start: 145
  - page_end: 147
  - has_images: true
  - image_paths: ["cache/images/page_145_img_2.png"]
  - has_tables: false
  - content_type: "procedural"
  - token_count: 342
```

### 3. Multi-Step Retrieval with Context Chaining

**What**: Retrieve for each sub-question sequentially, enhancing later queries with earlier context

**Why**: Complex questions have dependencies between sub-topics

**How**:
```python
context_chain = []

# Sub-question 1: "What is a no-code block?"
results_1 = retrieve(sub_q1)
context_chain.extend(results_1[:5])

# Sub-question 2: "How do I create a no-code block?"
# Enhanced with context from sub_q1
enhanced_q2 = enhance_with_context(sub_q2, context_chain)
results_2 = retrieve(enhanced_q2)
context_chain.extend(results_2[:5])

# Sub-question 3: "How do I test it with AFT?"
# Enhanced with context from sub_q1 + sub_q2
enhanced_q3 = enhance_with_context(sub_q3, context_chain)
results_3 = retrieve(enhanced_q3)
```

**Trade-off**: Sequential processing (slower) vs parallelization (breaks chaining)

**Current Choice**: Sequential for quality (can parallelize INDEPENDENT sub-questions in Phase 9)

### 4. Hybrid Search (Vector + BM25 + RRF)

**What**: Combine semantic search (embeddings) with keyword search (BM25)

**Why**:
- Vector search: Good for semantic similarity
- BM25: Good for exact keyword matches
- RRF: Optimal fusion method

**Parameters**:
- RRF k=60 (standard research value)
- Weights: 70% vector, 30% BM25
- Top-K: 50 per search method → 20 after reranking

**Tunability**: Can adjust weights based on query type in future

### 5. Strategy-Aware Generation

**What**: Adapt prompt and format based on query type

**Why**: Different questions need different answer styles

**Strategies**:
- Procedural → Step-by-step numbered list
- Comparison → Table format
- Troubleshooting → Problem/Solution structure
- Standard → Comprehensive paragraphs

**Impact**: Better formatting and user experience

### 6. Dataclass-Based Architecture

**What**: Use Python dataclasses for all data structures (NO ORM/database models)

**Why**:
- Type safety without ORM overhead
- Simple serialization to JSON
- Fast development
- No schema migrations

**Persistence**:
- Embeddings: Pickle files
- Chunks: JSON files
- Vector DB: Pinecone (cloud)
- BM25 Index: Pickle file

**Key Dataclasses**:
- `QueryUnderstanding` (Phase 3 output)
- `DecomposedQuery`, `SubQuestion` (decomposition)
- `RetrievalResult`, `OrganizedContext` (Phase 4 output)
- `GeneratedAnswer`, `ValidationResult` (Phase 6 output)
- `PipelineResult` (end-to-end output)

---

## Tech Stack

### Core Technologies

#### LLM & Embeddings
- **Groq** - LLM inference (Llama 3.3 70B, FREE tier)
  - Query decomposition (~1,000 tokens/query)
  - Answer generation (~6,000 tokens/query)
  - Rate limit: 100K tokens/day (~14 queries/day)

- **OpenAI** - Embeddings only
  - Model: `text-embedding-3-large` (3072-dim)
  - Cost: ~$0.0005 per query (3 sub-questions × 1 embedding each)
  - One-time indexing cost: ~$0.08 for 2,106 chunks

#### Vector Database & Search
- **Pinecone** - Vector database
  - Type: Serverless
  - Dimension: 3072
  - Metric: Cosine similarity
  - Index: 2,106 vectors
  - Free tier: 100K vectors, 1 index

- **rank-bm25** - Keyword search
  - BM25 algorithm implementation
  - Vocabulary: 16,460 terms
  - Index size: 64 MB (pickle)

#### Reranking
- **Cohere** - Semantic reranking
  - Model: `rerank-english-v3.0`
  - Cost: ~$0.0015 per query
  - Free tier: 1,000 requests/month

#### Document Processing
- **Docling** - PDF structure extraction
  - Heading hierarchy detection
  - Table extraction (HTML/Markdown)
  - Image extraction with captions
  - Better than PyMuPDF for structure preservation

#### UI & Web Framework
- **Streamlit** - Web interface
  - Real-time pipeline visualization
  - Metrics dashboard
  - Image gallery
  - Citation display
  - Caching with `@st.cache_resource`

### Supporting Libraries

#### Configuration & Validation
- **Pydantic** (v2) - Settings validation
  - Field validators for API keys
  - Type checking
  - Environment variable loading

- **python-dotenv** - Environment management

#### Text Processing
- **tiktoken** - Token counting (OpenAI tokenizer)
- **langchain** - Text splitting utilities
- **regex** - Advanced pattern matching

#### Data & Utilities
- **NumPy** - Numerical operations
- **Pillow** - Image processing
- **loguru** - Structured logging
- **tenacity** - Retry logic with exponential backoff

#### Development Tools
- **pytest** - Testing framework
- **black** - Code formatting
- **flake8** - Linting

---

## Folder & File Structure

### Root Directory

```
wm_help_assistant_2/
├── README.md                      # Project overview
├── CLAUDE.md                      # Claude Code guidance (1,300+ lines)
├── app.py                         # Streamlit web application
├── run_app.sh                     # Quick launcher script
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (API keys)
├── .env.example                   # Environment template
└── .gitignore                     # Git ignore rules
```

### Configuration (`config/`)

```
config/
└── settings.py                    # Pydantic settings class
    - API keys validation
    - Flat structure (not nested)
    - Field validators
    - validate_config() method
```

**Usage**:
```python
from config.settings import get_settings

settings = get_settings()
pdf_path = settings.pdf_path  # ✅ Flat access
# NOT: settings.paths.pdf_path  # ❌ Wrong
```

### Source Code (`src/`)

#### Ingestion Pipeline (`src/ingestion/`)

```
src/ingestion/
├── docling_processor.py           # PDF → Structured JSON
│   - Uses Docling library
│   - Extracts headings, tables, images
│   - Output: cache/docling_processed.json (43 MB)
│   - Time: ~15-60 min depending on hardware
│
├── hierarchical_chunker.py        # JSON → Context-Aware Chunks
│   - Section-based chunking
│   - Context injection (heading hierarchy)
│   - 20+ metadata fields per chunk
│   - Output: cache/hierarchical_chunks_filtered.json (4.5 MB, 2,106 chunks)
│   - Time: ~1-2 min
│
├── chunk_evaluator.py             # Chunk quality assessment
│   - Size consistency checks
│   - Structure preservation validation
│   - Context completeness scoring
│   - Output: tests/results/chunk_quality_report.txt
│
└── pymupdf_processor.py           # Alternative PDF processor (not used)
```

#### Query Understanding (`src/query/`)

```
src/query/
├── query_decomposer.py            # LLM-based query decomposition
│   - Groq Llama 3.3 70B
│   - Breaks complex queries into 2-4 sub-questions
│   - Identifies dependencies (INDEPENDENT, SEQUENTIAL, CONDITIONAL)
│   - Extracts topics per sub-question
│
├── query_expander.py              # Synonym-based query expansion (NEW)
│   - 32 synonym mappings
│   - 5 categories: actions, integrations, technical, concepts
│   - Generates 3 variations per query
│   - Major contributor to +42.8% recall
│
├── query_classifier.py            # Rule-based query classification
│   - Query types: procedural, conceptual, troubleshooting, etc.
│   - Query classes: simple, complex, multi-topic, etc.
│   - Pattern matching on keywords
│
├── intent_analyzer.py             # Intent extraction
│   - Intents: create, configure, integrate, troubleshoot, learn
│   - Keyword-based detection
│
├── query_understanding.py         # Orchestrator for Phase 3
│   - Combines decomposer, expander, classifier, intent analyzer
│   - Returns QueryUnderstanding dataclass
│
└── test_phase3.py                 # Phase 3 tests
```

#### Database & Indexing (`src/database/`)

```
src/database/
├── embedding_generator.py         # OpenAI embeddings generation
│   - Model: text-embedding-3-large (3072-dim)
│   - Two methods:
│     - generate_embeddings() for simple strings (queries)
│     - generate_embeddings_for_chunks() for chunk dicts
│   - Output: cache/hierarchical_embeddings.pkl (59 MB)
│
├── vector_store.py                # Pinecone vector database manager
│   - Create/delete index
│   - Upsert vectors in batches
│   - Query with metadata filtering
│   - Index: watermelon-docs-v2 (2,106 vectors)
│
├── bm25_index.py                  # BM25 keyword search index
│   - Uses rank-bm25 library
│   - Vocabulary: 16,460 terms
│   - Output: cache/bm25_index.pkl (64 MB)
│
└── run_phase5.py                  # Full Phase 5 pipeline
│   - Runs embedding generation + Pinecone upload + BM25 indexing
│   - One-time setup (~5-10 min)
```

#### Retrieval Pipeline (`src/retrieval/`)

```
src/retrieval/
├── hybrid_search.py               # Vector + BM25 + Query Expansion + RRF
│   - Vector search via Pinecone (top-50)
│   - BM25 search via rank-bm25 (top-50)
│   - Query expansion (3 variations per query)
│   - RRF fusion (k=60, 70/30 weights)
│   - Pinecone metadata limit workaround (3 maps)
│   - Returns deduplicated, scored results
│
├── reranker.py                    # Cohere semantic reranking
│   - Model: rerank-english-v3.0
│   - Input: Top 50 hybrid search results
│   - Output: Top 20 reranked results
│   - Cost: ~$0.0015 per call
│
├── context_organizer.py           # Result aggregation and organization
│   - Deduplication across sub-questions
│   - Score aggregation for multi-retrieved chunks
│   - Topic clustering
│   - Chronological ordering
│   - Output: OrganizedContext dataclass (20 chunks)
│
├── multi_step_retriever.py        # Multi-step retrieval orchestrator
│   - Sequential retrieval per sub-question
│   - Context chaining between steps
│   - Dependency handling (INDEPENDENT, SEQUENTIAL, CONDITIONAL)
│   - Returns RetrievalResult dataclass
│
└── test_phase4.py                 # Phase 4 tests
```

#### Generation Pipeline (`src/generation/`)

```
src/generation/
├── answer_generator.py            # LLM-based answer generation
│   - Groq Llama 3.3 70B
│   - 4 generation strategies (step-by-step, comparison, troubleshooting, standard)
│   - Multi-context integration (20 chunks)
│   - Citation extraction
│   - Image referencing
│   - Returns GeneratedAnswer dataclass
│
├── response_validator.py          # Quality validation
│   - Completeness check (all sub-questions answered?)
│   - Coherence scoring (logical flow?)
│   - Formatting validation (proper markdown?)
│   - Citation quality check
│   - Returns ValidationResult dataclass with overall score
│
└── end_to_end_pipeline.py         # Complete RAG pipeline
│   - Orchestrates all 4 stages
│   - Stage 1: Query Understanding
│   - Stage 2: Multi-Step Retrieval
│   - Stage 3: Answer Generation
│   - Stage 4: Validation
│   - Returns PipelineResult dataclass
```

#### Evaluation Framework (`src/evaluation/`)

```
src/evaluation/
├── retrieval_metrics.py           # IR metrics
│   - Precision@K
│   - Recall@K
│   - Mean Reciprocal Rank (MRR)
│   - Mean Average Precision (MAP)
│   - Normalized Discounted Cumulative Gain (NDCG)
│   - Coverage (% topics retrieved)
│   - Diversity (unique sections)
│
├── generation_metrics.py          # NLG metrics
│   - Overall quality score
│   - Completeness (all sub-questions answered?)
│   - Coherence (logical flow?)
│   - Formatting (proper markdown?)
│   - Word count
│
└── comprehensive_evaluation.py    # Batch evaluation script
│   - Tests N queries from tests/test_queries.json
│   - Computes all retrieval + generation metrics
│   - Output: tests/results/comprehensive_evaluation.json
│   - Interactive mode (asks how many queries to test)
```

#### Utilities (`src/utils/`)

```
src/utils/
└── toc_filter.py                  # Table of Contents filtering
    - Marks TOC chunks with is_toc=true flag
    - Filters TOC chunks during retrieval
    - Pages 1-18 are TOC in helpdocs.pdf
```

#### Memory (Unused) (`src/memory/`)

```
src/memory/
└── __init__.py                    # Empty - placeholder for future conversation memory
```

### Data Files (`data/`)

```
data/
├── helpdocs.pdf                   # Source PDF (150 MB, 2,257 pages)
└── helpdocs_test_50.pdf           # Test subset (93 MB, 50 pages)
```

### Cache Files (`cache/`)

```
cache/
├── docling_processed.json         # Structured document (43 MB)
├── hierarchical_chunks_filtered.json  # 2,106 chunks (4.5 MB)
├── hierarchical_embeddings.pkl    # Embeddings (59 MB)
├── bm25_index.pkl                 # BM25 index (64 MB)
└── images/                        # 1,454 extracted images (~68 KB total)
    ├── page_1_img_1.png
    ├── page_1_img_2.png
    └── ...
```

### Test Files (`tests/`)

```
tests/
├── test_queries.json              # 30 complex test queries
│   - Multi-topic procedural
│   - Multi-topic integration
│   - Conceptual + procedural
│   - Troubleshooting
│   - Security & compliance
│
└── results/
    ├── comprehensive_evaluation.json  # Full evaluation results (15 queries)
    ├── comprehensive_evaluation_BEFORE.json  # Before query expansion
    ├── comprehensive_evaluation_AFTER_EXPANSION.json  # After query expansion
    └── chunk_quality_report.txt  # Chunk quality metrics
```

### Documentation (`docs/`)

```
docs/
├── README.md                      # Documentation index
├── setup/
│   ├── getting-started.md         # Comprehensive setup guide
│   ├── setup.md                   # Basic setup
│   └── api-keys.md                # API key acquisition
├── guides/
│   ├── quick-start-ui.md          # UI usage guide
│   └── quality-improvement.md     # Troubleshooting guide
├── evaluation/
│   └── final-results.md           # Evaluation results
├── phases/
│   └── phase-8-ui.md              # Phase 8 completion
└── technical/
    ├── architecture.md            # THIS FILE - System architecture
    ├── ms-teams-fix.md            # Pinecone metadata fix
    └── toc-handling.md            # TOC filtering
```

### Utility Scripts (`scripts/`)

```
scripts/
└── compare_evaluations.py         # Compare two evaluation JSON files
    - Shows metric changes (Precision, Recall, MRR, etc.)
    - Color-coded improvements/regressions
    - Percentage changes
    - Usage: python scripts/compare_evaluations.py baseline.json new.json
```

---

## Data Flow

### Document Processing Pipeline (One-Time Setup)

```
┌─────────────────────────────────────────────────────────────────┐
│                     SOURCE PDF                                   │
│  data/helpdocs.pdf (150 MB, 2,257 pages)                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    python -m src.ingestion.docling_processor
                    (~15-60 min, CPU-intensive)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               STRUCTURED DOCUMENT (Docling Output)               │
│  cache/docling_processed.json (43 MB)                           │
│  + cache/images/*.png (1,454 images)                            │
│                                                                  │
│  Structure:                                                      │
│  - Heading hierarchy (H1→H2→H3→H4)                             │
│  - Tables (HTML/Markdown)                                       │
│  - Images (PNG with captions)                                   │
│  - Cross-references                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                python -m src.ingestion.hierarchical_chunker
                    (~1-2 min)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              HIERARCHICAL CHUNKS (Context-Aware)                 │
│  cache/hierarchical_chunks_filtered.json (4.5 MB, 2,106 chunks) │
│                                                                  │
│  Each chunk:                                                     │
│  - Content with prepended section hierarchy                     │
│  - 20+ metadata fields (heading_path, images, tables, etc.)    │
│  - Token count, character count                                 │
│  - is_toc flag (for TOC filtering)                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                python -m src.database.embedding_generator
                    (~5 min, $0.08 cost)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   EMBEDDINGS (OpenAI)                            │
│  cache/hierarchical_embeddings.pkl (59 MB)                      │
│                                                                  │
│  - 2,106 vectors × 3072 dimensions                             │
│  - Model: text-embedding-3-large                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
            ┌─────────────────┴─────────────────┐
            ↓                                   ↓
   python -m src.database.vector_store   python -m src.database.bm25_index
         (~2 min)                              (~1 min)
            ↓                                   ↓
┌─────────────────────────┐         ┌─────────────────────────┐
│   PINECONE INDEX        │         │   BM25 INDEX            │
│  (Cloud Vector DB)      │         │  cache/bm25_index.pkl   │
│                         │         │  (64 MB)                │
│  - 2,106 vectors        │         │                         │
│  - 3072-dim             │         │  - 16,460 vocab terms   │
│  - Cosine similarity    │         │  - rank-bm25 algorithm  │
│  - Serverless           │         │                         │
└─────────────────────────┘         └─────────────────────────┘
```

### Query Processing Pipeline (Per Query)

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER QUERY                                  │
│  "How do I create a no-code block and test it with AFT?"       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                 ╔═══════════════════════════════════╗
                 ║   STAGE 1: QUERY UNDERSTANDING    ║
                 ╚═══════════════════════════════════╝
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                     ↓                      ↓
  Query Decomposer      Query Expander       Query Classifier
  (Groq LLM)           (32 synonyms)         (Rule-based)
        ↓                     ↓                      ↓
  4 sub-questions      3 variations/query    "multi-topic_procedural"
  with dependencies    per sub-question       + intents
        └─────────────────────┬─────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│             QUERY UNDERSTANDING OUTPUT                           │
│  - Sub-questions: 4                                             │
│  - Query variations: 3 per sub-question                         │
│  - Query class: "multi-topic_procedural"                        │
│  - Intents: ["create", "test", "integrate"]                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                 ╔═══════════════════════════════════╗
                 ║   STAGE 2: MULTI-STEP RETRIEVAL   ║
                 ╚═══════════════════════════════════╝
                              ↓
        FOR EACH SUB-QUESTION (sequential with context chaining):
                              ↓
        ┌──────────────────────────────────────────┐
        │  Sub-question 1: "What is a no-code      │
        │  block in Watermelon?"                   │
        └──────────────────────────────────────────┘
                              ↓
                    Query Expansion (3 variations)
                              ↓
            ┌─────────────────┴─────────────────┐
            ↓                                   ↓
    Vector Search                         BM25 Search
    (Pinecone)                           (rank-bm25)
    Top-50 results                       Top-50 results
            └─────────────────┬─────────────────┘
                              ↓
                    RRF Fusion (k=60, 70/30 weights)
                              ↓
                    Cohere Reranking (top-20)
                              ↓
        ┌──────────────────────────────────────────┐
        │  20 relevant chunks for sub-question 1   │
        │  + Add top 5 to context_chain            │
        └──────────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────────┐
        │  Sub-question 2: "How do I create a      │
        │  no-code block?"                         │
        │  (Enhanced with context_chain from sq1)  │
        └──────────────────────────────────────────┘
                              ↓
                    [Same process: Expand → Search → Fuse → Rerank]
                              ↓
        ┌──────────────────────────────────────────┐
        │  20 relevant chunks for sub-question 2   │
        │  + Add top 5 to context_chain            │
        └──────────────────────────────────────────┘
                              ↓
                    [Repeat for sub-questions 3 & 4]
                              ↓
                    Context Organizer
                    (Deduplicate + Cluster + Order)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              ORGANIZED CONTEXT OUTPUT                            │
│  - Final chunks: 20 (max)                                       │
│  - Topics covered: ["no-code blocks", "testing", "AFT"]         │
│  - Images: 8 relevant images                                    │
│  - Page ranges: [45-52, 89-94, 123-128]                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                 ╔═══════════════════════════════════╗
                 ║   STAGE 3: ANSWER GENERATION      ║
                 ╚═══════════════════════════════════╝
                              ↓
            Strategy Selection ("step-by-step" for procedural)
                              ↓
            Multi-Context Prompting (20 chunks)
                              ↓
            Groq LLM Generation (Llama 3.3 70B)
                              ↓
            Citation Extraction + Image Referencing
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 GENERATED ANSWER                                 │
│                                                                  │
│  # How to Create a No-Code Block and Test with AFT              │
│                                                                  │
│  ## Step 1: Create a No-Code Block                              │
│  1. Navigate to Workflows > No-Code Blocks                      │
│  2. Click "Create New Block"                                    │
│  ...                                                             │
│                                                                  │
│  ## Step 2: Configure for Testing                               │
│  ...                                                             │
│                                                                  │
│  **Citations**: [Section: No-Code Blocks (p. 45-52)]           │
│  **Images**: [No-Code Block Interface.png]                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                 ╔═══════════════════════════════════╗
                 ║   STAGE 4: VALIDATION             ║
                 ╚═══════════════════════════════════╝
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                     ↓                      ↓
  Completeness Check    Coherence Check      Formatting Check
  (All sub-qs?)         (Logical flow?)      (Proper markdown?)
        └─────────────────────┬─────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 VALIDATION RESULT                                │
│  - Overall score: 0.92 (Excellent)                              │
│  - Completeness: 1.00 (All sub-questions answered)             │
│  - Coherence: 0.95                                              │
│  - Formatting: 0.88                                             │
│  - Citation quality: 0.85                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   FINAL PIPELINE RESULT                          │
│                                                                  │
│  - Query: [original]                                            │
│  - Query Understanding: [Stage 1 output]                        │
│  - Retrieval Result: [Stage 2 output]                           │
│  - Generated Answer: [Stage 3 output]                           │
│  - Validation: [Stage 4 output]                                 │
│  - Metrics: {retrieval: {...}, generation: {...}}              │
│  - Timing: {total: 27.7s, per_stage: {...}}                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Display in Streamlit UI
```

---

## Design Patterns

### 1. Dataclass-Based Data Flow

**Pattern**: All data structures are Python dataclasses (NO ORM/database models)

**Rationale**:
- Type safety without ORM overhead
- Simple JSON serialization via `asdict()`
- Fast development, no migrations
- Explicit structure

**Implementation**:
```python
from dataclasses import dataclass, field, asdict
from typing import List, Optional

@dataclass
class SubQuestion:
    id: str
    question: str
    topics: List[str] = field(default_factory=list)
    priority: int = 1

# Serialization
result = SubQuestion(id="q1", question="What is X?", topics=["topic1"])
result_dict = asdict(result)  # Recursively converts to dict
json.dump(result_dict, f)
```

**Critical Rules**:
- Use `field(default_factory=list)` for mutable defaults (NOT `= []`)
- Access nested fields directly: `result.answer.answer` (NOT `result.answer_text`)
- Always use `asdict()` for JSON serialization

### 2. Synchronous Processing (Intentional)

**Pattern**: No async/await anywhere in the codebase

**Rationale**:
- Simpler code and debugging
- Context chaining requires sequential processing
- Easier to understand execution flow

**Trade-off**: Speed vs simplicity (currently favoring simplicity)

**Future Optimization**:
- Parallelize INDEPENDENT sub-questions using `ThreadPoolExecutor`
- Keep SEQUENTIAL/CONDITIONAL sub-questions sequential
- Estimated speedup: 40-50%

### 3. Three-Map Pinecone Recovery Pattern

**Pattern**: Maintain content_map + metadata_map + embeddings in memory

**Problem**: Pinecone 40KB metadata limit prevents storing full chunk content/metadata

**Solution**:
```python
# Load all data at initialization
self.chunk_content_map = {chunk['metadata']['chunk_id']: chunk['content']}
self.chunk_metadata_map = {chunk['metadata']['chunk_id']: chunk['metadata']}
self.chunk_embeddings = {chunk['metadata']['chunk_id']: chunk['embedding']}

# During retrieval: Merge Pinecone results with full data
chunk_id = match.metadata['chunk_id']
content = self.chunk_content_map.get(chunk_id, '')  # Full content!
full_metadata = self.chunk_metadata_map.get(chunk_id, {})
merged = {**match.metadata, **full_metadata}
```

**Impact**: Fixed ALL integration queries (MS Teams, Shopify, Slack)

### 4. Strategy Pattern for Generation

**Pattern**: Select generation strategy based on query type

**Implementation**:
```python
def _get_generation_strategy(self, query_class: str) -> str:
    if "procedural" in query_class:
        return "step-by-step"
    elif "comparison" in query_class:
        return "comparison"
    elif "troubleshooting" in query_class:
        return "troubleshooting"
    else:
        return "standard"
```

**Benefits**:
- Better formatting per query type
- Improved user experience
- Extensible (easy to add new strategies)

### 5. Graceful Degradation Pattern

**Pattern**: Try/except with fallbacks for optional dependencies

**Implementation**:
```python
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("⚠️  Docling not installed. Please run: pip install docling")
    DocumentConverter = None

# Later in code
if DocumentConverter is None:
    raise RuntimeError("Docling library not installed")
```

**Benefits**:
- Clear error messages
- Doesn't fail on import
- Guides users to solution

### 6. Pydantic Settings Pattern

**Pattern**: Centralized configuration with validation

**Implementation**:
```python
from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    openai_api_key: str

    @field_validator('openai_api_key')
    def validate_api_key(cls, v):
        if not v or "xxx" in v.lower():
            raise ValueError("Invalid API key")
        return v

# Usage
from config.settings import get_settings
settings = get_settings()  # Validates on load
```

**Benefits**:
- Type safety
- Automatic validation
- Environment variable loading
- Clear error messages

---

## Performance Characteristics

### Query Processing Time Breakdown

**Average Total**: 27.7 seconds per query

| Stage | Time | % of Total |
|-------|------|------------|
| Query Understanding | ~2-3s | 10% |
| Multi-Step Retrieval | ~18-20s | 70% |
| Answer Generation | ~4-5s | 16% |
| Validation | ~1s | 4% |

**Bottlenecks**:
1. **Sequential sub-question retrieval** (70% of time)
   - Solution: Parallelize INDEPENDENT sub-questions
   - Potential speedup: 40-50%

2. **Multiple API calls per sub-question**
   - 3 query variations × 2 search methods = 6 searches per sub-question
   - Cohere reranking adds latency
   - Solution: Batch API calls, use async

3. **LLM generation** (Groq free tier)
   - Already very fast (FREE tier!)
   - Groq Pro API would be slightly faster

### Memory Usage

| Component | Memory |
|-----------|--------|
| Embeddings (in memory) | ~230 MB |
| BM25 Index (in memory) | ~80 MB |
| Chunk content/metadata maps | ~15 MB |
| Streamlit cache | ~50 MB |
| **Total** | **~375 MB** |

**Note**: Lightweight for a RAG system

### Cost Per Query

| Service | Cost |
|---------|------|
| OpenAI (3 query embeddings) | $0.0005 |
| Cohere (3 reranking calls) | $0.0015 |
| Groq (decomposition + generation) | $0.0000 (FREE) |
| Pinecone (vector queries) | $0.0000 (FREE tier) |
| **Total** | **~$0.002** |

**Monthly (300 queries)**: ~$0.60

### Scalability Considerations

**Current System**:
- Handles 2,106 chunks comfortably
- 2,257 pages of documentation
- ~14 queries/day (Groq free tier limit)

**Scaling to 10K+ Pages**:
- Vector DB: Pinecone scales to millions of vectors (OK)
- BM25 Index: May need optimization for large vocabularies
- Memory: May need to lazy-load embeddings
- Retrieval time: May increase linearly with corpus size

**Scaling to 1000+ Queries/Day**:
- Need Groq Pro API (or switch to OpenAI/Anthropic)
- Need Cohere paid tier
- Consider Redis caching for common queries
- Parallelize retrieval

---

## Summary

This RAG system combines:
- **Hierarchical document processing** (structure preservation)
- **Query expansion** (32 synonym mappings, +42.8% recall)
- **Multi-step retrieval** (context chaining, hybrid search, reranking)
- **Strategy-aware generation** (4 different strategies)
- **Comprehensive validation** (quality scoring)

**Production-Ready**: 89% complete (Phases 1-8), Phase 9 (deployment) pending

**Next Steps**: See [docs/README.md](../README.md) for getting started

---

**Last Updated**: November 4, 2025
**Version**: Phase 8 Complete
**Status**: Production-Ready (pending deployment)
