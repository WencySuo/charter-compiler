# Agent Compiler Technical Specification
## Orchestration-Level Memoization for LLM Inference

### Executive Summary
This specification defines the architecture for an agent compiler that leverages orchestration-level memoization to optimize LLM inference performance. The system analyzes Python agent code, builds execution DAGs, identifies memoization opportunities, and generates optimal TensorRT-LLM cache configurations to outperform vLLM's standard cache optimization on Lambda Labs bare metal GPUs with Triton.

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Agent Source Code                        │
│                        (main.py)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    AST Parser Module                         │
│              (ast + libcst for preservation)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 DAG Construction Module                      │
│            (NetworkX + Scalpel for CFG analysis)            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               Memoization Analysis Engine                    │
│         (Pattern detection & optimization planning)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Cache Config Generator                          │
│        (TensorRT-LLM specific optimizations)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Execution Orchestrator                       │
│          (Triton client with cache management)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Performance Monitor & Profiler                  │
│            (Prometheus metrics + custom tracking)            │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. AST Parser Module (`charter_compiler/parser/`)

**Purpose**: Parse Python agent code and extract LLM call patterns

**Key Classes**:
- `AgentASTParser`: Main parser using Python's `ast` module
- `LLMCallExtractor`: Identifies LLM inference calls in AST
- `DependencyTracker`: Tracks variable dependencies between calls

**Data Structures**:
```python
@dataclass
class LLMCall:
    id: str
    prompt_template: str
    input_variables: List[str]
    output_variable: str
    line_number: int
    dependencies: List[str]  # IDs of dependent LLM calls
    
@dataclass
class ParsedAgent:
    calls: List[LLMCall]
    control_flow: ControlFlowGraph
    data_flow: DataFlowGraph
```

### 2. DAG Construction Module (`charter_compiler/dag/`)

**Purpose**: Build execution DAG from parsed AST

**Key Classes**:
- `DAGBuilder`: Constructs NetworkX DiGraph from parsed agent
- `ControlFlowAnalyzer`: Uses Scalpel to extract CFG
- `DataFlowAnalyzer`: Tracks data dependencies
- `LoopDetector`: Identifies loop patterns for optimization

**Key Algorithms**:
- Topological sorting for execution order
- Strongly connected components for loop detection
- Common subexpression elimination for shared prompts

### 3. Memoization Analysis Engine (`charter_compiler/analysis/`)

**Purpose**: Identify memoization opportunities and patterns

**Key Classes**:
- `MemoizationAnalyzer`: Main analysis orchestrator
- `PrefixAnalyzer`: Finds shared prompt prefixes
- `SequencePatternDetector`: Identifies sequential chains
- `BranchingAnalyzer`: Detects branching patterns
- `LoopOptimizer`: Optimizes loop-based patterns

**Pattern Types**:
```python
class MemoizationPattern(Enum):
    SHARED_PREFIX = "shared_prefix"        # Multiple calls with same prefix
    SEQUENTIAL_CHAIN = "sequential_chain"  # A->B->C dependency
    BRANCHING = "branching"                # A->(B|C|D)
    LOOP_INVARIANT = "loop_invariant"     # Constant within loop
    DELTA_MATCHING = "delta_matching"     # Small variations
```

**Analysis Metrics**:
- Prefix overlap ratio (target: >60%)
- Sequence depth (optimal: 3-7 steps)
- Branch factor (optimal: 2-4)
- Loop iteration count
- Delta size (% of prompt changed)

### 4. Cache Config Generator (`charter_compiler/config/`)

**Purpose**: Generate optimal TensorRT-LLM configurations

**Key Classes**:
- `TRTLLMConfigGenerator`: Main config generator
- `CacheStrategySelector`: Chooses caching strategy
- `PriorityMapper`: Maps patterns to cache priorities
- `BlockSizeOptimizer`: Selects optimal block size

**Configuration Templates**:
```python
@dataclass
class CacheConfig:
    enable_kv_cache_reuse: bool = True
    tokens_per_block: int = 32  # 32 for high reuse, 128 for efficiency
    kv_cache_free_gpu_mem_fraction: float = 0.95
    batch_scheduler_policy: str = "max_utilization"
    retention_configs: List[RetentionConfig] = None
    
@dataclass
class RetentionConfig:
    start_token: int
    end_token: int
    priority: int  # 0-100
    duration: int  # seconds
```

### 5. Execution Orchestrator (`charter_compiler/executor/`)

**Purpose**: Execute compiled agent with optimized caching

**Key Classes**:
- `TritonOrchestrator`: Manages Triton client connections
- `CacheIDManager`: Assigns and tracks cache IDs
- `ExecutionScheduler`: Orders operations for cache reuse
- `RequestBatcher`: Groups requests for efficiency

**Execution Strategies**:
- Sequential execution with cache warming
- Parallel branching with shared prefix
- Loop unrolling with delta tracking

### 6. Performance Monitor (`charter_compiler/monitor/`)

**Purpose**: Track and compare performance vs baseline

**Key Classes**:
- `MetricsCollector`: Collects Prometheus metrics
- `PerformanceProfiler`: Tracks latency/throughput
- `CacheHitAnalyzer`: Monitors cache reuse rates
- `BaselineComparator`: Compares to vLLM baseline

**Metrics Tracked**:
- Time-to-first-token (TTFT)
- Time-per-output-token (TPOT)
- Cache hit ratio
- Memory utilization
- Request throughput

## Test Case Specifications

### Test Case 1: Sequential Chain (Extraction → Validation → Classification)

**Description**: Multi-stage document processing pipeline

**Implementation**:
```python
class DocumentProcessor:
    def process(self, document: str) -> ProcessingResult:
        # Stage 1: Entity Extraction
        entities = self.llm_extract_entities(
            prompt=f"{SYSTEM_PROMPT}\nExtract entities from:\n{document}"
        )
        
        # Stage 2: Validation (depends on extraction)
        validated = self.llm_validate_entities(
            prompt=f"{SYSTEM_PROMPT}\nValidate these entities:\n{entities}\nAgainst document:\n{document}"
        )
        
        # Stage 3: Classification (depends on validation)
        classification = self.llm_classify_document(
            prompt=f"{SYSTEM_PROMPT}\nClassify document:\n{document}\nWith validated entities:\n{validated}"
        )
        
        return ProcessingResult(entities, validated, classification)
```

**Memoization Opportunities**:
- SYSTEM_PROMPT shared across all stages (100 tokens)
- Document context reused in stages 2-3 (500-1000 tokens)
- Progressive context accumulation (entities → validated)

**Expected Optimization**:
- 60-70% cache reuse on stages 2-3
- 40% latency reduction overall
- TTFT reduction from 2s to 0.8s for stages 2-3

### Test Case 2: Delta Matching Loop

**Description**: Iterative processing with small input variations

**Implementation**:
```python
class DeltaProcessor:
    def batch_process(self, template: str, variables: List[Dict]) -> List[str]:
        results = []
        
        # Shared context across all iterations
        base_prompt = f"{SYSTEM_PROMPT}\n{template}"
        
        for i, vars in enumerate(variables):
            # Only variable substitution changes
            full_prompt = base_prompt.format(**vars)
            result = self.llm_generate(prompt=full_prompt)
            results.append(result)
            
        return results
```

**Variable Examples**:
```python
variables = [
    {"company": "TechCorp", "revenue": "$10M", "employees": "50"},
    {"company": "DataInc", "revenue": "$15M", "employees": "75"},
    {"company": "CloudCo", "revenue": "$8M", "employees": "40"},
    # ... 10-20 variations
]
```

**Memoization Opportunities**:
- Base template shared (90% of prompt)
- Only variable substitutions differ (10%)
- Loop iterations share prefix

**Expected Optimization**:
- 80-90% cache reuse after first iteration
- 5-10x throughput improvement
- Memory efficiency from block sharing

## Implementation Phases

### Phase 1A: Core Infrastructure (Week 1)

1. **Package Structure Setup**
   - Create `charter_compiler` package
   - Set up `pyproject.toml` with dependencies
   - Configure logging and error handling
   - Set up pytest infrastructure

2. **Basic AST Parser**
   - Implement `AgentASTParser` class
   - Create `LLMCallExtractor` visitor
   - Build simple dependency tracker
   - Unit tests for parsing

3. **Simple DAG Builder**
   - NetworkX integration
   - Basic topological sorting
   - Cycle detection
   - Visualization utilities

### Phase 1B: Analysis Engine (Week 2)

4. **Pattern Detection**
   - Implement `PrefixAnalyzer`
   - Build `SequencePatternDetector`
   - Create `LoopOptimizer`
   - Pattern matching unit tests

5. **Config Generation**
   - TensorRT-LLM config templates
   - Priority mapping logic
   - Block size selection
   - Config validation

6. **Basic Orchestrator**
   - Triton client wrapper
   - Cache ID management
   - Sequential execution
   - Error handling

### Phase 1C: Test Implementation (Week 3)

7. **Test Case 1 Implementation**
   - Document processor agent
   - Benchmark harness
   - Baseline measurements
   - Performance tracking

8. **Test Case 2 Implementation**
   - Delta processor agent
   - Loop optimization
   - Batch processing
   - Comparative analysis

9. **Performance Profiling**
   - Metrics collection
   - Visualization dashboard
   - Comparison reports
   - Optimization recommendations

## Configuration Files

### `charter_compiler/configs/default.yaml`
```yaml
parser:
  preserve_comments: false
  track_line_numbers: true
  
dag:
  max_depth: 10
  detect_cycles: true
  
memoization:
  min_prefix_length: 50  # tokens
  prefix_overlap_threshold: 0.6
  enable_loop_unrolling: true
  max_loop_iterations: 100
  
cache:
  backend: "tensorrt-llm"
  tokens_per_block: 32
  gpu_memory_fraction: 0.95
  enable_quantization: false  # FP8
  
execution:
  max_batch_size: 8
  sequential_delay_ms: 50
  enable_warmup: true
  
monitoring:
  prometheus_port: 8002
  enable_profiling: true
  log_level: "INFO"
```

### `charter_compiler/configs/benchmarks.yaml`
```yaml
baselines:
  vllm:
    enable_prefix_caching: true
    gpu_memory_utilization: 0.90
    block_size: 16
    
  tensorrt_llm_base:
    enable_kv_cache_reuse: false  # No optimization
    
test_cases:
  sequential_chain:
    iterations: 100
    document_size: 1000  # tokens
    warmup_iterations: 5
    
  delta_matching:
    iterations: 20
    variations: 10
    base_template_size: 500  # tokens
    delta_size: 50  # tokens
```

## Success Metrics

### Performance Targets
- **Latency Reduction**: 40-60% vs vLLM baseline
- **Throughput Improvement**: 2-3x for batch workloads
- **Cache Hit Rate**: >70% for sequential chains, >85% for delta matching
- **Memory Efficiency**: 30% reduction in GPU memory usage
- **TTFT**: <1s for cached prompts (vs 2-3s baseline)

### Validation Criteria
- Correctness: Output equivalence to baseline
- Stability: <5% variance across runs
- Scalability: Linear scaling to 8 concurrent requests
- Reliability: Zero cache corruption errors

## Dependencies

### Core Requirements
```python
# pyproject.toml dependencies
[tool.poetry.dependencies]
python = "^3.10"
ast = "*"  # Built-in
libcst = "^1.0.0"
networkx = "^3.0"
scalpel = "^0.2.0"
numpy = "^1.24.0"
tritonclient = {extras = ["grpc", "http"], version = "^2.40.0"}
prometheus-client = "^0.19.0"
pyyaml = "^6.0"
click = "^8.1.0"
rich = "^13.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
pytest-benchmark = "^4.0.0"
black = "^23.0.0"
ruff = "^0.1.0"
mypy = "^1.7.0"

[tool.poetry.group.viz.dependencies]
matplotlib = "^3.7.0"
graphviz = "^0.20.0"
plotly = "^5.18.0"
```

## Risk Mitigation

### Technical Risks
1. **Cache Timing Issues**: Implement retry logic with exponential backoff
2. **Memory Overflow**: Monitor usage, implement gradual eviction
3. **Pattern Misidentification**: Conservative defaults, manual override options
4. **Version Compatibility**: Pin all dependencies, test matrix

### Performance Risks
1. **Compilation Overhead**: Cache compiled configs
2. **Analysis Bottleneck**: Limit DAG depth, use heuristics
3. **Network Latency**: Local Triton deployment, connection pooling

## Next Steps

1. Set up repository structure
2. Implement basic AST parser
3. Create test case templates
4. Establish baseline measurements
5. Build incremental optimizations
6. Validate against benchmarks