# Enhanced Cache - Quick Reference Card

## 📁 Essential Files (Your Cache Folder)

```
✅ USE THESE FILES:
├── hierarchical_chunks_enhanced_final.json  (5.37 MB) ← Your main data
├── hierarchical_embeddings.pkl              (60 MB)   ← Pre-built embeddings
├── bm25_index.pkl                          (8.3 MB)  ← Hybrid search
└── images_enhanced/                        (1,549)   ← Renamed images

🔐 BACKUPS (safe to ignore):
└── backup/hierarchical_chunks_enriched.json          ← Original backup
```

---

## 🎯 One-Line Loaders

```python
# Load chunks
import json
with open('cache/hierarchical_chunks_enhanced_final.json') as f:
    chunks = json.load(f)['chunks']

# Load embeddings
import pickle
with open('cache/hierarchical_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Load BM25 index
with open('cache/bm25_index.pkl', 'rb') as f:
    bm25 = pickle.load(f)
```

---

## 🔍 Essential Queries

### By Content Type (Semantic)
```python
# Find troubleshooting content
troubleshoot = [c for c in chunks if c['metadata']['content_type'] == 'troubleshooting']

# Find configuration guides
configs = [c for c in chunks if c['metadata']['content_type'] == 'configuration']

# Find integration docs
integrations = [c for c in chunks if c['metadata']['content_type'] == 'integration']
```

### By Features (Structural)
```python
# Find code examples
code = [c for c in chunks if c['metadata'].get('has_code')]

# Find chunks with tables
tables = [c for c in chunks if c['metadata'].get('has_tables')]

# Find chunks with images
images = [c for c in chunks if c['metadata'].get('image_paths')]
```

### By Topic (AI-Enhanced)
```python
# Find by topic
kubernetes = [c for c in chunks if 'kubernetes' in c['metadata'].get('topics', [])]
security = [c for c in chunks if 'security' in c['metadata'].get('topics', [])]
automation = [c for c in chunks if 'automation' in c['metadata'].get('topics', [])]
```

### Combined Filters (Power Queries)
```python
# Kubernetes config with code
k8s_config_code = [
    c for c in chunks
    if c['metadata']['content_type'] == 'configuration'
    and 'kubernetes' in c['metadata'].get('topics', [])
    and c['metadata'].get('has_code')
]

# Troubleshooting with images
troubleshoot_visual = [
    c for c in chunks
    if c['metadata']['content_type'] == 'troubleshooting'
    and c['metadata'].get('image_paths')
]
```

---

## 📊 Quick Stats

```python
print(f"Total chunks: {len(chunks)}")
print(f"With code: {sum(1 for c in chunks if c['metadata'].get('has_code'))}")
print(f"With tables: {sum(1 for c in chunks if c['metadata'].get('has_tables'))}")
print(f"With images: {sum(1 for c in chunks if c['metadata'].get('image_paths'))}")
print(f"With topics: {sum(1 for c in chunks if c['metadata'].get('topics'))}")
```

---

## 🏷️ Available Metadata Fields

```python
{
    # Identification
    'chunk_id': 'sec_156_chunk_2',
    'section_id': 'sec_156',

    # Pages
    'page_start': 38,
    'page_end': 40,

    # Content Type (Semantic)
    'content_type': 'configuration',  # troubleshooting, integration, etc.

    # AI Enhancements
    'topics': ['kubernetes', 'deployment'],
    'content_summary': 'How to deploy...',
    'integration_names': ['Slack', 'MS Teams'],

    # Structure Detection
    'has_code': True,
    'has_tables': False,
    'has_images': True,
    'has_lists': True,

    # Images
    'image_paths': ['cache/images_enhanced/k8s_config_chunk156_page38_img0.png'],
    'image_captions': ['Deployment diagram'],

    # Hierarchy
    'heading_path': ['Getting Started', 'Configuration'],
    'current_heading': 'Kubernetes Setup',
    'heading_level': 2,

    # Size
    'token_count': 245,
    'char_count': 1230,
}
```

---

## 🎨 Content Types (Semantic Classification)

| Type | Count | Use For |
|------|-------|---------|
| `troubleshooting` | 499 | Error resolution, debugging |
| `integration` | 477 | Tool setup, API configs |
| `general` | 382 | Overview, concepts |
| `configuration` | 310 | Settings, setup guides |
| `procedural` | 206 | Step-by-step tutorials |
| `security` | 177 | Auth, permissions, SSL |
| `conceptual` | 82 | Theory, architecture |

---

## 🔖 Common Topics

```python
topics = [
    'automation', 'testing', 'kubernetes', 'docker',
    'security', 'ai', 'users', 'deployment',
    'monitoring', 'ci/cd', 'api', 'database'
]
```

---

## 🖼️ Image Naming Convention

```
{section_name}_chunk{section_id}_page{page_num}_img{index}.png

Examples:
• active_directory_integration_chunksec_18_chunk_0_page55_img0.png
• kubernetes_config_chunksec_156_page38_img0.png
• slack_integration_chunksec_89_page67_img0.png
```

**Benefits:**
- Searchable by topic
- Linked to specific chunk
- Page reference included
- Multiple images indexed

---

## 🚀 RAG Pipeline Quick Start

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load chunks
with open('cache/hierarchical_chunks_enhanced_final.json') as f:
    chunks = json.load(f)['chunks']

# Convert to LangChain docs
docs = [Document(page_content=c['content'], metadata=c['metadata']) for c in chunks]

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name='watermelon_enhanced'
)

# Query with metadata filters
results = vectorstore.similarity_search(
    "How to configure Slack integration?",
    k=5,
    filter={'content_type': 'integration'}
)
```

---

## 💡 Pro Tips

### 1. Use Semantic Filters First
```python
# ✅ Good: Filter then search
troubleshoot = [c for c in chunks if c['metadata']['content_type'] == 'troubleshooting']
results = search_in(troubleshoot, query)

# ❌ Bad: Search everything
results = search_in(chunks, query)
```

### 2. Combine Multiple Filters
```python
# Multi-criteria search
results = [
    c for c in chunks
    if c['metadata']['content_type'] == 'configuration'  # Semantic
    and c['metadata'].get('has_code')                   # Structural
    and 'kubernetes' in c['metadata'].get('topics', []) # Topic
]
```

### 3. Use Content Summaries
```python
# Get quick overview
for chunk in results:
    print(chunk['metadata'].get('content_summary', chunk['content'][:200]))
```

### 4. Leverage Integration Names
```python
# Find all Slack-related content
slack_content = [
    c for c in chunks
    if 'slack' in str(c['metadata'].get('integration_names', [])).lower()
]
```

---

## 📈 Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Load chunks | ~1s | 2,133 chunks, 5.37 MB |
| Load embeddings | ~2s | 60 MB pickle file |
| Filtered search | <10ms | In-memory filtering |
| Vector search | 50-100ms | With embeddings |
| Hybrid search | 100-150ms | Vector + BM25 |

---

## 🔧 Troubleshooting

**Q: Import error when loading pickle?**
```python
# Make sure you have the right libraries
pip install scikit-learn numpy
```

**Q: Image paths not working?**
```python
# Update to use enhanced images
old_path = "cache/images/page_0038_img_00.png"
new_path = "cache/images_enhanced/kubernetes_config_chunksec_156_page38_img0.png"
```

**Q: Want to see all available content types?**
```python
from collections import Counter
types = Counter(c['metadata']['content_type'] for c in chunks)
print(types)
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `QUICK_START_ENHANCED.md` | Code examples & tutorials |
| `COMPARISON_REPORT.md` | Cache vs Friend analysis |
| `ENHANCEMENT_GUIDE.md` | How enhancement was done |
| `REFERENCE_CARD.md` | This file - quick reference |

---

## 🎯 Success Checklist

- [ ] Can load enhanced chunks JSON
- [ ] Can filter by content_type
- [ ] Can filter by topics
- [ ] Can detect code chunks
- [ ] Can access renamed images
- [ ] Can use pre-built embeddings
- [ ] Ready to build RAG pipeline!

---

**🏆 You have the best RAG data possible - start building!**

**Questions?** Check:
1. `QUICK_START_ENHANCED.md` for detailed examples
2. `COMPARISON_REPORT.md` for feature breakdown
3. Your cache folder for all the data
