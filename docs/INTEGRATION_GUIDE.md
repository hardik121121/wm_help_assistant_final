# Integration Guide: Enhanced Chunks + RAG Pipeline

This guide explains how the `docling_processor` enhanced chunks have been integrated with the `mw_help_assistant_2` RAG pipeline.

## What Was Done

### 1. Data Integration ✅

The `integrate_with_rag.py` script has successfully:

- **Loaded 2,133 enhanced chunks** from `cache/hierarchical_chunks_enhanced_final.json`
- **Adapted format** for RAG pipeline compatibility
- **Copied 1,549 semantically-named images** to RAG cache
- **Updated image paths** to match new directory structure
- **Preserved all AI-enhanced metadata**:
  - AI topics (95.4% of chunks)
  - Content summaries (100% of chunks)
  - Semantic types (troubleshooting, integration, configuration, etc.)
  - Code/table/image detection flags
  - Integration names
  - Technical depth classification

### 2. Statistics

Your integrated dataset:

- **Total chunks**: 2,133
- **AI-enhanced**: 2,034 chunks with topics (95.4%)
- **Code snippets**: 329 chunks (15.4%)
- **Images**: 719 chunks reference images (33.7%)
- **Tables**: 6 chunks (0.3%)
- **Average size**: 282 tokens per chunk
- **Coverage**: 2,257 pages

**Content type distribution**:
- Troubleshooting: 499 (23.4%)
- Integration: 477 (22.4%)
- General: 382 (17.9%)
- Configuration: 310 (14.5%)
- Procedural: 206 (9.7%)
- Security: 177 (8.3%)
- Conceptual: 82 (3.8%)

### 3. Files Created

```
mw_help_asistant_2/
├── cache/
│   ├── hierarchical_chunks.json              # Enhanced chunks (primary)
│   ├── hierarchical_chunks_filtered.json     # Enhanced chunks (alias)
│   ├── integration_stats.json                # Integration statistics
│   └── images/                               # 1,549 semantically-named images
```

## Next Steps to Run the RAG Pipeline

### Step 1: Navigate to RAG Directory

```bash
cd mw_help_asistant_2
```

### Step 2: Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

**Required API Keys**:
1. **OpenAI** (embeddings): https://platform.openai.com/api-keys
2. **Pinecone** (vector DB): https://app.pinecone.io/
3. **Cohere** (re-ranking): https://dashboard.cohere.com/api-keys
4. **Groq** (LLM): https://console.groq.com/keys

### Step 3: Install Dependencies (if not done)

```bash
# Create virtual environment (if not exists)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Generate Embeddings and Indexes

This will create:
- Vector embeddings using OpenAI (text-embedding-3-large)
- Pinecone vector index
- BM25 keyword index

```bash
python -m src.database.run_phase5
```

⏱️ **Expected time**: ~5-10 minutes
💰 **Cost**: ~$0.50 (one-time for 2,133 chunks)

### Step 5: Launch the RAG Pipeline

```bash
# Quick launch with script
./run_app.sh

# Or manually
streamlit run app.py
```

The application will open at `http://localhost:8501`

## What You Get

### Enhanced RAG System Features

Your RAG pipeline now benefits from the enhanced chunks:

#### 1. **Better Retrieval**
- **AI topics** enable semantic filtering (e.g., find all "kubernetes" + "security" chunks)
- **Content types** help filter by intent (troubleshooting vs integration vs config)
- **Code/table flags** identify technical content
- **Integration names** improve matching for tool-specific queries

#### 2. **Richer Context**
- **Content summaries** provide quick overviews
- **Technical depth** classification helps match user expertise
- **Semantic image names** make visual context more discoverable

#### 3. **Smarter Search**
Example query capabilities:
```python
# Find troubleshooting chunks with code examples
filter: content_type=troubleshooting AND has_code=true

# Find Slack integration guides with images
filter: integration_names contains "Slack" AND has_images=true

# Find high-depth configuration content
filter: content_type=configuration AND technical_depth=high
```

### RAG Pipeline Architecture

```
User Query
    ↓
Query Understanding (LLM-based decomposition)
    ↓
Multi-Step Retrieval
  ├─ Vector Search (Pinecone)
  ├─ BM25 Keyword Search
  └─ Reciprocal Rank Fusion (RRF)
    ↓
Cohere Reranking (semantic precision)
    ↓
Context Organization (topic clustering)
    ↓
Answer Generation (Groq Llama 3.3 70B)
    ↓
Response Validation
    ↓
Comprehensive Answer with Citations
```

## Advantages Over Standard Pipeline

### Your Enhanced System vs Standard RAG

| Feature | Standard RAG | Your Enhanced System |
|---------|-------------|---------------------|
| Chunk metadata | Basic (page, section) | 23 fields including AI topics, summaries |
| Content classification | None | 7 semantic types |
| Code detection | Manual | Automatic (15.4% detected) |
| Integration awareness | Keyword-based | Named entity extraction |
| Image organization | Generic names | Semantic naming |
| Technical depth | Not classified | Low/Medium/High |
| Query expansion | Basic | 32 synonym mappings + AI topics |

### Performance Expectations

Based on `mw_help_assistant_2` benchmarks:

**Retrieval Quality**:
- Precision@10: **0.667** (67% of results relevant)
- Recall@10: **0.638** (finds 64% of relevant content)
- MRR: **0.854** (top result highly ranked)

**Generation Quality**:
- Overall Score: **0.914** (91.4% quality)
- Completeness: **1.000** (100% - all sub-questions answered)
- Quality Distribution: **100% Excellent** (all queries ≥0.85)

**Performance**:
- Average query time: **27.7s**
- Cost per query: **$0.003**

## Usage Examples

### Example 1: Multi-Topic Query

**Query**: "How do I create a no-code block on Watermelon and process it for Autonomous Functional Testing?"

**What happens**:
1. Query decomposed into 4 sub-questions
2. Hybrid search finds relevant chunks by topic tags
3. Reranking prioritizes chunks with `content_type=procedural` and `has_code=true`
4. Context organized by topics: ["no-code blocks", "testing", "workflow"]
5. Answer generated with step-by-step instructions
6. Images automatically included from enhanced metadata

### Example 2: Integration Query

**Query**: "What are the integration steps for MS Teams and how do I configure automated responses?"

**What happens**:
1. Query classifier identifies: `multi-topic_integration`
2. Retrieval filters chunks with `integration_names=["MS Teams"]`
3. Combines `content_type=integration` + `content_type=configuration` chunks
4. Organizes by section hierarchy
5. Generates comprehensive guide with citations

## Troubleshooting

### Issue: "No module named 'src'"

**Solution**: Always use `python -m src.module.name` format

```bash
# ✅ Correct
python -m src.database.run_phase5

# ❌ Wrong
python src/database/run_phase5.py
```

### Issue: Groq Rate Limit Exceeded

**Solution**: Free tier = 14 queries/day. Test with 5 queries first.

```python
# In comprehensive_evaluation.py, enter: 5
```

### Issue: Empty Retrieval Results

**Solution**: Verify embeddings and indexes were generated:

```bash
ls -lh mw_help_asistant_2/cache/
# Should show:
# - hierarchical_embeddings.pkl (~60 MB)
# - bm25_index.pkl (~8 MB)
```

### Issue: Missing Images in Answers

**Solution**: Verify images were copied:

```bash
ls mw_help_asistant_2/cache/images/ | wc -l
# Should show: 1549
```

## Advanced Configuration

### Tuning Retrieval (in .env)

```bash
# Increase coverage (more chunks retrieved)
VECTOR_TOP_K=75  # default: 50
BM25_TOP_K=75    # default: 50

# Increase final results (more context for generation)
RERANK_TOP_K=30  # default: 20

# Adjust hybrid search balance
RRF_K=60         # default: 60 (range: 20-100)
```

### Tuning Generation (in .env)

```bash
# Control answer length
LLM_MAX_TOKENS=8192      # default: 8192

# Control creativity vs consistency
LLM_TEMPERATURE=0.2      # default: 0.2 (range: 0-1)

# Enable/disable features
ENABLE_QUERY_DECOMPOSITION=true  # default: true
MAX_SUB_QUESTIONS=4              # default: 4
```

## Monitoring & Evaluation

### Run Comprehensive Evaluation

```bash
cd mw_help_asistant_2
python -m src.evaluation.comprehensive_evaluation
# Enter number of queries (start with 5)
```

This measures:
- **Retrieval metrics**: Precision, Recall, MRR, MAP, NDCG
- **Generation metrics**: Completeness, Quality, Formatting
- **Performance**: Time, cost, success rate

### View Results

```bash
cat tests/results/comprehensive_evaluation.json
```

### Compare Before/After Changes

```bash
# Save baseline
cp tests/results/comprehensive_evaluation.json tests/results/baseline.json

# Make changes, re-run evaluation

# Compare
python scripts/compare_evaluations.py \
  tests/results/baseline.json \
  tests/results/comprehensive_evaluation.json
```

## Cost Breakdown

### One-Time Setup
- **Embeddings** (2,133 chunks): ~$0.50
- **Total**: **~$0.50**

### Per Query
- OpenAI embedding: $0.0001
- Cohere reranking: $0.002
- Groq LLM: $0 (free tier)
- **Total per query**: **~$0.003**

### Monthly (300 queries)
- **~$10-15**

### Free Tier Limits
- Groq: 14,400 requests/day (14 complex queries/day)
- Pinecone: 100,000 vectors (you have 2,133)
- Cohere: 1,000 calls/month (then $0.002/call)

## Benefits Summary

### What You Gain with Enhanced Chunks

1. **95.4% Topic Coverage** - AI-extracted topics enable semantic search
2. **100% Summarization** - Every chunk has a summary for quick context
3. **15.4% Code Detection** - Automatic identification of code snippets
4. **33.7% Image Coverage** - Chunks know their visual context
5. **7 Semantic Types** - Better query routing and filtering
6. **Semantic Image Names** - Easier to understand visual content

### Unique Advantages

- **Multi-topic queries** handled better due to topic tags
- **Integration queries** improved with named entity extraction
- **Code-heavy questions** benefit from `has_code` flag
- **Visual context** enhanced by semantic image naming
- **Technical depth** matching improves user experience

## Files Reference

### Input (from docling_processor)
- `cache/hierarchical_chunks_enhanced_final.json` - Enhanced chunks (5.37 MB)
- `cache/images_enhanced/` - Semantically named images (1,549 files)
- `cache/hierarchical_embeddings.pkl` - Pre-built embeddings (60 MB) - NOT USED
- `cache/bm25_index.pkl` - Pre-built BM25 index (8.3 MB) - NOT USED

*Note: Pre-built embeddings/indexes not used because RAG pipeline uses different embedding model (OpenAI vs local)*

### Output (in mw_help_assistant_2)
- `cache/hierarchical_chunks.json` - Adapted chunks for RAG
- `cache/hierarchical_chunks_filtered.json` - Same (alias)
- `cache/images/` - Copied images
- `cache/hierarchical_embeddings.pkl` - Generated by Phase 5
- `cache/bm25_index.pkl` - Generated by Phase 5
- `cache/integration_stats.json` - Integration statistics

## Next Actions

1. ✅ **Integration complete** - Enhanced chunks ready
2. ⏳ **Set up .env** - Add your API keys
3. ⏳ **Run Phase 5** - Generate embeddings and indexes
4. ⏳ **Launch app** - Start querying!

## Support

- **RAG Architecture**: `mw_help_asistant_2/docs/technical/architecture.md`
- **RAG Setup**: `mw_help_asistant_2/docs/setup/getting-started.md`
- **Enhanced Chunks Guide**: `REFERENCE_CARD.md`
- **Quick Start**: `QUICK_START_ENHANCED.md`

---

**Built for Maximum Quality. Designed for Complex Queries. Optimized for Production.**
