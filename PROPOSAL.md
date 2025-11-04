# REfrag Implementation Proposal

## Executive Summary

This document outlines the implementation plan for **REfrag (REpresentation For RAG)**, a novel retrieval-augmented generation optimization framework introduced in the paper "REFRAG: Rethinking RAG based Decoding" (arXiv:2509.01092). The implementation will focus on creating a production-ready, end-to-end system in Python that demonstrates the core compress → sense/select → expand methodology without building LLMs or embedding models from scratch.

## Paper Overview

### Problem Statement

Traditional RAG systems suffer from:
- **High latency**: Processing long concatenated passages introduces significant Time-To-First-Token (TTFT) delays
- **Memory inefficiency**: Large key-value caches for long contexts reduce throughput
- **Wasted computation**: Only a small subset of retrieved passages is typically relevant to the query
- **Sparse attention patterns**: Retrieved passages often have low semantic similarity, leading to block-diagonal attention patterns

### REfrag Solution

REfrag addresses these issues through a three-phase approach:

1. **COMPRESS**: Split retrieved passages into fixed-size chunks (e.g., 16 tokens) and encode each into dense embeddings using a lightweight encoder
2. **SENSE/SELECT**: Use a policy (RL-based or heuristic) to identify information-dense chunks that should bypass compression
3. **EXPAND**: Feed the LLM a hybrid input combining compressed chunk embeddings (projected to token space) and raw tokens for critical chunks

### Key Benefits

- **30× TTFT acceleration** compared to traditional RAG
- **16× context window expansion** beyond standard models
- **No accuracy loss** compared to full-token baselines
- **Compatible with existing LLMs** without modification

## Architecture Design

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        REfrag System                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. RETRIEVAL LAYER                                          │
│     ├─ Document Store (Vector DB: ChromaDB)                 │
│     ├─ Embedding Model (sentence-transformers)              │
│     └─ Query Handler                                         │
│                                                               │
│  2. COMPRESSION LAYER                                        │
│     ├─ Text Chunker (configurable chunk size)               │
│     ├─ Chunk Encoder (RoBERTa-based from transformers)      │
│     └─ Embedding Cache                                       │
│                                                               │
│  3. PROJECTION LAYER                                         │
│     ├─ Projector Network (Linear layer)                     │
│     └─ Token-space Mapper                                    │
│                                                               │
│  4. SELECTION LAYER (Policy)                                 │
│     ├─ Heuristic Policy (PPL-based, TF-IDF, BM25)          │
│     ├─ Learned Policy (optional: simple RL)                 │
│     └─ Chunk Scorer                                          │
│                                                               │
│  5. GENERATION LAYER                                         │
│     ├─ LLM Interface (OpenAI API / Anthropic / HuggingFace)│
│     ├─ Hybrid Input Constructor                             │
│     └─ Response Handler                                      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Python Version**: 3.10+
- **Package Manager**: `uv`
- **Core Libraries**:
  - `transformers` - For encoder models (RoBERTa) and tokenizers
  - `sentence-transformers` - For document embeddings
  - `chromadb` - Vector database for document retrieval
  - `openai` - LLM API (primary choice)
  - `anthropic` - Alternative LLM API
  - `torch` - For neural network components
  - `numpy` - Numerical operations
  - `tiktoken` - Token counting for metrics

- **Development Tools**:
  - `jupyter` - For demonstration notebook
  - `matplotlib` / `seaborn` - Visualization
  - `python-dotenv` - Environment management
  - `pytest` - Testing framework
  - `black` / `ruff` - Code formatting and linting

## Implementation Strategy

### Phase 1: Core Infrastructure (Days 1-2)

**Objective**: Set up project structure and basic components

- Initialize Python project with `uv`
- Define configuration management system
- Implement document store and vector DB integration
- Create basic retrieval mechanism
- Set up API interfaces for LLM and embeddings

**Deliverables**:
- Project structure with proper modules
- Configuration files (.env, config.yaml)
- Basic retrieval working end-to-end

### Phase 2: Compression Pipeline (Days 3-4)

**Objective**: Implement the COMPRESS phase

- Text chunking utility (configurable size)
- Chunk encoder using pre-trained RoBERTa
- Embedding cache mechanism for efficiency
- Batch processing for multiple chunks
- Token counting and metrics

**Deliverables**:
- Working compression module
- Unit tests for chunking and encoding
- Performance benchmarks

### Phase 3: Projection & Selection (Days 5-6)

**Objective**: Implement PROJECTION and SENSE/SELECT phases

- Projector network to map embeddings to token space
- Heuristic-based selection policies:
  - Perplexity-based scoring
  - TF-IDF importance
  - Semantic similarity to query
  - BM25 ranking
- Configurable expansion fraction (how many chunks to expand)
- Hybrid input constructor

**Deliverables**:
- Projection module with trainable/fixed mappings
- Multiple selection strategies
- Configuration for policy selection

### Phase 4: Generation Pipeline (Days 7-8)

**Objective**: Integrate with LLM and complete EXPAND phase

- LLM interface supporting multiple providers
- Hybrid input formatting (compressed + raw tokens)
- Response generation and handling
- End-to-end pipeline integration
- Error handling and fallbacks

**Deliverables**:
- Complete REfrag pipeline
- Support for OpenAI/Anthropic/HuggingFace APIs
- Graceful degradation to standard RAG

### Phase 5: Demonstration & Benchmarking (Days 9-10)

**Objective**: Create comprehensive demo and documentation

- Jupyter notebook with end-to-end examples
- Comparative benchmarks:
  - Token usage: REfrag vs Standard RAG
  - Time metrics: TTFT, total latency
  - Quality: Answer accuracy/relevance
- Visualization of results
- Performance profiling

**Deliverables**:
- Interactive demonstration notebook
- Benchmark results and visualizations
- Performance analysis

### Phase 6: Documentation (Day 11)

**Objective**: Comprehensive documentation

- README.md with:
  - Clear installation instructions
  - Quick start guide
  - Architecture overview
  - API documentation
  - Configuration guide
  - Examples
- Code comments and docstrings
- Architecture diagrams

**Deliverables**:
- Professional README.md
- Well-commented codebase
- Usage examples

## Simplified Implementation Decisions

To focus on the core methodology while avoiding complexity:

### 1. Pre-trained Models Only
- **Encoder**: Use pre-trained `roberta-base` from HuggingFace
- **Embeddings**: Use pre-trained `all-MiniLM-L6-v2` or `bge-small-en-v1.5`
- **LLM**: Use API-based models (OpenAI GPT-4, Claude, or HuggingFace Inference)
- **No training**: Skip continual pre-training (CPT) phase from paper

### 2. Heuristic Selection vs Full RL
- **Start with heuristics**: Implement multiple selection strategies
  - Perplexity-based (simulate importance)
  - TF-IDF scoring
  - Query-chunk similarity
  - Hybrid scoring
- **Optional RL**: Can be added later if needed
- **Configurable**: Easy to switch between strategies

### 3. Simplified Projector
- **Linear projection**: Simple learnable or fixed linear layer
- **Dimension matching**: Project encoder embeddings to match LLM token dimension
- **Pre-computed**: Can use fixed random projection initially
- **Extensible**: Architecture allows for more complex projectors

### 4. API-First LLM Integration
- **Benefit**: No need to manage model weights or GPU memory
- **Trade-off**: Can't directly manipulate token embeddings
- **Solution**: Use prompt-based approach where compressed chunks are represented as special tokens or summarized text
- **Fallback**: For full implementation, support HuggingFace models with direct embedding manipulation

### 5. Focus on Demonstration
- **Skip evaluation tasks**: No need for benchmark datasets from paper
- **Custom demos**: Create clear examples showing benefits
- **Metrics focus**: Token reduction, time savings, cost reduction

## Project Structure

```
refrag/
├── README.md                 # Comprehensive documentation
├── PROPOSAL.md              # This file
├── pyproject.toml           # uv project configuration
├── .env                     # API keys (not committed)
├── .env_example             # Template for environment variables
├── .gitignore              # Git ignore rules
│
├── refrag/                  # Main package
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── retrieval/          # Retrieval and vector DB
│   │   ├── __init__.py
│   │   ├── vector_store.py  # ChromaDB integration
│   │   ├── embedder.py      # Embedding generation
│   │   └── retriever.py     # Query and retrieve
│   │
│   ├── compression/        # Compression phase
│   │   ├── __init__.py
│   │   ├── chunker.py      # Text chunking
│   │   ├── encoder.py      # Chunk encoding
│   │   └── cache.py        # Embedding cache
│   │
│   ├── projection/         # Projection phase
│   │   ├── __init__.py
│   │   └── projector.py    # Embedding projection
│   │
│   ├── selection/          # Selection phase
│   │   ├── __init__.py
│   │   ├── base.py         # Base policy class
│   │   ├── heuristic.py    # Heuristic policies
│   │   └── learned.py      # Optional RL policy
│   │
│   ├── generation/         # Generation phase
│   │   ├── __init__.py
│   │   ├── llm_interface.py # LLM API abstraction
│   │   └── hybrid_input.py  # Hybrid input constructor
│   │
│   ├── pipeline.py         # End-to-end pipeline
│   └── utils.py            # Utilities and helpers
│
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_retrieval.py
│   ├── test_compression.py
│   ├── test_projection.py
│   ├── test_selection.py
│   └── test_pipeline.py
│
├── notebooks/              # Demonstrations
│   └── demo.ipynb         # End-to-end demonstration
│
├── examples/              # Example scripts
│   ├── basic_usage.py
│   ├── benchmark.py
│   └── compare_rag.py
│
└── data/                  # Sample data
    ├── documents/         # Sample documents
    └── queries/           # Sample queries
```

## Key Implementation Details

### 1. Text Chunking

```python
def chunk_text(text: str, chunk_size: int = 16, tokenizer) -> List[str]:
    """
    Split text into fixed-size token chunks.

    Args:
        text: Input text to chunk
        chunk_size: Number of tokens per chunk (default: 16)
        tokenizer: Tokenizer for counting tokens

    Returns:
        List of text chunks
    """
    # Tokenize text
    # Split into chunks of chunk_size
    # Convert back to text
    # Handle edge cases (incomplete chunks)
```

### 2. Chunk Encoding

```python
class ChunkEncoder:
    """
    Encodes text chunks into dense embeddings using RoBERTa.
    """
    def __init__(self, model_name: str = "roberta-base"):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, chunks: List[str]) -> np.ndarray:
        """
        Encode chunks into embeddings using [CLS] token pooling.

        Returns:
            Array of shape (num_chunks, embedding_dim)
        """
        # Batch encode chunks
        # Extract [CLS] embeddings
        # Return as numpy array
```

### 3. Projector Network

```python
class Projector(nn.Module):
    """
    Projects chunk embeddings to LLM token embedding space.
    """
    def __init__(self, encoder_dim: int, llm_dim: int):
        super().__init__()
        self.projection = nn.Linear(encoder_dim, llm_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project encoder embeddings to token space.
        """
        return self.projection(embeddings)
```

### 4. Selection Policy

```python
class HeuristicPolicy:
    """
    Selects important chunks based on heuristics.
    """
    def __init__(self, strategy: str = "similarity", top_k: int = 4):
        self.strategy = strategy
        self.top_k = top_k

    def select(self, chunks: List[str], query: str,
               embeddings: np.ndarray) -> List[int]:
        """
        Select top-k most important chunk indices.

        Strategies:
        - 'similarity': Cosine similarity to query
        - 'tfidf': TF-IDF importance scores
        - 'hybrid': Combination of multiple signals

        Returns:
            List of indices of selected chunks
        """
```

### 5. REfrag Pipeline

```python
class REfragPipeline:
    """
    End-to-end REfrag pipeline.
    """
    def __init__(self, config: Config):
        self.retriever = Retriever(config)
        self.encoder = ChunkEncoder(config)
        self.projector = Projector(config)
        self.policy = Policy(config)
        self.llm = LLMInterface(config)

    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Process query through full REfrag pipeline.

        Returns:
            {
                'answer': Generated answer,
                'metrics': {
                    'tokens_saved': int,
                    'compression_ratio': float,
                    'ttft': float,
                    'total_time': float
                }
            }
        """
        # 1. Retrieve relevant documents
        # 2. Chunk and encode (COMPRESS)
        # 3. Select important chunks (SENSE/SELECT)
        # 4. Construct hybrid input (EXPAND)
        # 5. Generate answer with LLM
        # 6. Collect metrics
```

## API Key Requirements

The implementation will require API keys for:

- **OpenAI API** (primary LLM choice): For GPT-4 or GPT-3.5-turbo
- **Anthropic API** (alternative): For Claude models
- **HuggingFace** (optional): For inference API or model downloads

Environment variables in `.env`:
```bash
# LLM API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
HUGGINGFACE_API_KEY=your_hf_key_here

# Configuration
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-3.5-turbo
CHUNK_SIZE=16
EXPANSION_FRACTION=0.25
```

## Success Criteria

The implementation will be considered successful if it:

1. **Functional Completeness**:
   - ✅ Implements all three phases (compress, sense/select, expand)
   - ✅ Supports end-to-end query-to-answer pipeline
   - ✅ Works with standard LLM APIs
   - ✅ Handles error cases gracefully

2. **Performance Demonstration**:
   - ✅ Shows measurable token reduction (target: 4-16×)
   - ✅ Shows time savings in TTFT (target: 2-5× in API context)
   - ✅ Maintains answer quality comparable to standard RAG

3. **Code Quality**:
   - ✅ Well-structured, modular codebase
   - ✅ Comprehensive comments and docstrings
   - ✅ Follows Python best practices (PEP 8, type hints)
   - ✅ Unit tests for core components
   - ✅ Configurable and extensible design

4. **Documentation**:
   - ✅ Clear README with installation and usage
   - ✅ Working demonstration notebook
   - ✅ Architecture explanation
   - ✅ Example use cases

5. **Usability**:
   - ✅ Simple installation via `uv`
   - ✅ Clear configuration options
   - ✅ Example scripts provided
   - ✅ Easy to integrate into existing projects

## Demonstration Notebook Contents

The `demo.ipynb` notebook will include:

1. **Setup and Installation**
   - Environment setup
   - API key configuration
   - Sample data loading

2. **Basic Usage**
   - Simple query example
   - Standard RAG comparison
   - Output visualization

3. **Compression Benefits**
   - Token usage comparison
   - Compression ratio analysis
   - Visualization of chunk selection

4. **Performance Benchmarks**
   - TTFT measurements
   - Total latency comparison
   - Cost analysis (API tokens)

5. **Quality Assessment**
   - Answer comparison (REfrag vs Standard RAG)
   - Relevance scoring
   - Example outputs

6. **Configuration Exploration**
   - Effect of chunk size
   - Effect of expansion fraction
   - Different selection strategies

## Risk Mitigation

### Technical Risks

1. **API Rate Limits**:
   - Mitigation: Add rate limiting, caching, retry logic

2. **Embedding Dimension Mismatch**:
   - Mitigation: Use configurable projection layer, dimension adaptation

3. **Selection Quality**:
   - Mitigation: Implement multiple heuristics, allow strategy selection

4. **Integration Complexity**:
   - Mitigation: Abstract LLM interface, support multiple providers

### Scope Risks

1. **Over-complexity**:
   - Mitigation: Focus on core methodology, skip RL training initially

2. **Time Constraints**:
   - Mitigation: Prioritized implementation phases, MVP first

## Future Extensions

After core implementation, potential extensions include:

1. **Advanced Selection**:
   - Implement RL-based policy training
   - Multi-objective optimization (accuracy + efficiency)

2. **Additional LLM Support**:
   - Local model support (Llama, Mistral)
   - Direct embedding manipulation for better integration

3. **Performance Optimization**:
   - GPU acceleration
   - Embedding caching strategies
   - Batch processing improvements

4. **Enhanced Retrieval**:
   - Hybrid search (dense + sparse)
   - Re-ranking integration
   - Multi-hop retrieval

5. **Evaluation Framework**:
   - Benchmark datasets from paper
   - Automated quality metrics
   - A/B testing framework

## Conclusion

This implementation plan provides a pragmatic approach to building REfrag from scratch while leveraging existing tools and APIs. The focus is on demonstrating the core compress-sense/select-expand methodology and its benefits in a production-ready, extensible codebase. By avoiding the complexity of training LLMs and embedding models, we can focus on the novel contribution of REfrag: efficient representation and selective expansion for RAG systems.

The implementation will serve as both a reference implementation and a practical tool for improving RAG system efficiency in real-world applications.

---

**Estimated Timeline**: 11 days
**Lines of Code**: ~2,000-3,000 (excluding tests and examples)
**Test Coverage Target**: >80%
