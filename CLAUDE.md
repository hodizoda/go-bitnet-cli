# go-bitnet

CGO Go bindings for BitNet 1-bit LLM inference with native LangChainGo provider.

## Build

Prerequisites: cmake >= 3.22, clang >= 18, python >= 3.9, huggingface-cli

    make libs          # Build bitnet.cpp static libraries
    make models        # Download default test model (2B4T)
    make test          # Build + download + run tests
    make bench MODEL=path/to/model.gguf  # Benchmark a model

## Architecture

- `bitnet.go` — CGO bindings wrapping llama.h from bitnet.cpp
- `options.go` — Functional options for model/context/LLM configuration
- `langchain.go` — LangChainGo llms.Model adapter
- `3rdparty/BitNet/` — git submodule (microsoft/BitNet)
- `build/` — cmake output (static libs, gitignored)
- `models/` — GGUF model files (gitignored)

## Conventions

- Package name: `bitnet`
- Test with: `go test -v ./...` (requires built libs + downloaded model)
- CGO links statically against libllama.a + libggml.a
- All C memory managed via explicit Close() methods (idempotent)
- Thread safety: Model is shareable, Context is single-goroutine

## Gotchas

- CGO flags in bitnet.go reference 3rdparty/ paths — don't move files
- bitnet.cpp kernel codegen requires python 3.9+ and clang 18+
- llama_token_eos/bos deprecated — use llama_vocab_eos/bos
- llama_tokenize returns negative n if buffer too small (negate for required size)
- Batch fields (token, pos, seq_id, logits) are C arrays — use unsafe.Slice to access
