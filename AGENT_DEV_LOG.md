### Agent Dev Log

Date: 2025-09-30

Purpose: Record design decisions, assumptions, and concerns while implementing from `TECHNICAL_SPEC.md` and `IMPLEMENTATION_TASKS.md`.

Decisions and Notes:

- AST extraction scope: Full prompt reconstruction via AST (including variable resolution and f-strings) is complex. For POC, we implemented a heuristic parser that:
  - Detects `self.llm.generate(...)` calls
  - Establishes sequential dependencies between consecutive calls in a function
  - Marks calls inside loops to infer delta-matching patterns
  This enables detection of `SEQUENTIAL_CHAIN` and `DELTA_MATCHING` patterns without full prompt introspection.

- Tokenization: The spec mentions token-level prefix matching. We provided a simple whitespace tokenizer for prefix analysis as a placeholder. Real deployments should plug a model-accurate tokenizer (e.g., SentencePiece/BPE) to align with cache block boundaries.

- Triton integration: Actual IO tensor names depend on the deployed model config (`config.pbtxt`). We used plausible names (`text_input`, `text_output`, `prompt_table_extra_id`, etc.). In this environment, Triton is not available, so the orchestrator simulates metrics for tests. Replace with real Triton calls in production and validate tensor names against the model.

- Config generator: Followed the spec for TensorRT-LLM parameters and retention configs. Priority mapping is heuristic and should be tuned with empirical data.

- Dependencies: `scalpel` is listed but not used in this POC. It can be integrated later for precise control/data flow analysis. `pydantic` was included (per tasks) but not currently used.

Risks / Follow-ups:

- The AST heuristic may miss non-standard LLM invocation patterns. Future work: richer pattern library and CFG/DFG-based dependency extraction.
- Metrics are simulated; end-to-end numbers must be validated on real hardware with Triton and a TensorRT-LLM engine implementing prompt table support.
- Prefix analysis currently ignores true tokenization and BPE boundaries; swap in the target model tokenizer.

Environment blockers:

- The CI environment lacks Python execution (`python: command not found`), so tests couldn't be executed here. Code structure and lints pass in static checks; run `poetry install && pytest` in a Python 3.10 environment to execute tests.


