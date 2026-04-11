# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ms-swift (ModelScope SWIFT) is a large-scale LLM/multimodal model fine-tuning and deployment framework. It supports 500+ models, 150+ built-in datasets, and multiple training methods (SFT, RLHF, pre-training, DPO, KTO, etc.) with various distributed training backends.

## Common Commands

### Install
```bash
pip install -e .                    # Editable install from source
pip install -e '.[eval]'            # With evaluation dependencies
pip install -e '.[all]'             # All optional dependencies
```

### Lint & Format
```bash
pre-commit run --all-files          # Run all linting (flake8, isort, yapf, etc.)
make linter                         # Alternative via Makefile
```

### Test
```bash
make test                                                          # Full CI test suite
python tests/run.py --parallel 2 --run_config tests/run_config.yaml  # Parallel test runner
python -m pytest tests/utils/                                       # Run a specific test directory
python -m pytest tests/utils/test_xxx.py                            # Run a single test file
```

### Build
```bash
make whl                            # Build wheel: python setup.py sdist bdist_wheel
make docs                           # Build Sphinx documentation
```

### CLI Usage
The `swift` CLI dispatches to subcommands:
```bash
swift sft --model <model> --dataset <dataset>          # Supervised fine-tuning
swift pt --model <model> --dataset <dataset>            # Pre-training
swift rlhf --model <model> --dataset <dataset>          # RLHF training
swift infer --model <model>                             # Interactive inference
swift deploy --model <model>                            # Deploy as OpenAI-compatible API
swift export --model <model>                            # Export/quantize model
swift eval --model <model>                              # Evaluate model
swift merge-lora --model <model> --adapters <path>      # Merge LoRA weights
swift web-ui                                            # Launch Gradio web UI
swift app --model <model>                               # Launch chat app
```

Multi-GPU via environment variables (auto-wraps with `torchrun`):
```bash
NPROC_PER_NODE=4 swift sft --model <model> --dataset <dataset>
```

Arguments can be passed via YAML/JSON config files:
```bash
swift sft config.yaml
```

There is also a `megatron` CLI for Megatron distributed training.

## Code Style

- Max line length: 120 characters
- Formatter: yapf (PEP8 based)
- Import sorting: isort (multi_line_output=0)
- Linter: flake8 (ignores: F401, F403, F405, F821, W503, E251, W504, E126)
- Strings: double quotes are converted to single quotes by pre-commit hook
- Line endings: LF enforced

## Architecture

### Main Package (`swift/`)

**Entry flow**: CLI (`swift.cli.main`) routes subcommands via `ROUTE_MAPPING` to individual CLI modules, which call pipeline `*_main()` functions.

**Core modules**:

- **`arguments/`** — Dataclass-based argument definitions (training, inference, export, eval). These are the central configuration objects.
- **`pipelines/`** — High-level orchestrators (`sft_main`, `infer_main`, `deploy_main`, `eval_main`, `export_main`, etc.). Each pipeline wires together arguments, model loading, dataset preparation, and execution.
- **`model/`** — Model registration, loading, and processing. Models are registered with metadata (architecture, template mapping, supported features). Uses a registry pattern.
- **`template/`** — Chat/prompt templates for each model family. Templates handle tokenization, chat formatting, and multimodal input processing. Critical for correct model I/O.
- **`dataset/`** — Dataset loading, preprocessing, and registration. Supports HuggingFace datasets, ModelScope datasets, and custom formats.
- **`trainers/`** — Extends HuggingFace `Trainer` and `Seq2SeqTrainer` with SWIFT-specific features (loss computation, callback integration, etc.).
- **`rlhf_trainers/`** — Specialized trainers for RLHF algorithms (DPO, KTO, GRPO, etc.), built on top of the `trl` library.
- **`tuners/`** — Parameter-efficient fine-tuning methods (LoRA, etc.), extending the `peft` library.
- **`infer_engine/`** — Inference backend abstraction with implementations for vLLM, SGLang, LmDeploy, and native Transformers.
- **`megatron/`** — Megatron-LM integration for large-scale distributed training (tensor/pipeline/sequence parallelism).
- **`ui/`** — Gradio-based web interface for training and inference.

**Utility patterns**:
- Lazy module loading via `_LazyModule` in `__init__.py` files — imports are deferred until accessed.
- Extensive use of `@register_*` decorators for models, datasets, and templates.
- Environment variable-driven distributed training configuration (`NPROC_PER_NODE`, `NNODES`, etc.).
