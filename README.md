# go-bitnet

CGO Go bindings for [BitNet](https://github.com/microsoft/BitNet) 1-bit LLM inference, with a native [LangChainGo](https://github.com/tmc/langchaingo) provider.

Run 1-bit quantized LLMs (BitNet b1.58) on CPU with no GPU required. Models use ternary weights ({-1, 0, 1}), enabling fast inference on commodity hardware.

## Features

- **Low-level API** -- Load models, tokenize, decode, sample, and generate completions directly
- **LangChainGo integration** -- Drop-in `llms.Model` implementation for use in LangChain pipelines
- **Streaming** -- Token-by-token streaming with stop string detection and context cancellation
- **Multi-turn chat** -- ChatML template formatting with conversation history
- **CLI tool** -- Interactive chat and benchmarking modes

## Supported Models

Any GGUF model quantized with BitNet's `i2_s` format. Tested with:

| Model | Parameters | Size on disk |
|-------|-----------|-------------|
| [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) | 2B | ~500 MB |
| [Falcon3-3B-Instruct](https://huggingface.co/tiiuae/Falcon3-3B-Instruct-1.58bit-GGUF) | 3B | ~750 MB |
| [Falcon3-7B-Instruct](https://huggingface.co/tiiuae/Falcon3-7B-Instruct-1.58bit-GGUF) | 7B | ~1.5 GB |
| [Falcon3-10B-Instruct](https://huggingface.co/tiiuae/Falcon3-10B-Instruct-1.58bit-GGUF) | 10B | ~2.2 GB |
| [Llama3-8B-1.58bit](https://huggingface.co/huyrua1996/Llama3-8B-1.58-100B-tokens-bitnet) | 8B | ~1.7 GB |

## Prerequisites

- Go 1.22+
- CMake >= 3.22
- Clang >= 18
- Python >= 3.9 (for kernel codegen)
- `huggingface_hub` Python package (`pip install huggingface_hub`)

## Quick Start

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/hodizoda/go-bitnet-cli.git
cd go-bitnet-cli

# Build static libraries (bitnet.cpp + llama.cpp)
make libs

# Download a model
make models  # downloads bitnet-b1.58-2B-4T (~500 MB)

# Run the CLI chat
go run ./cmd/bitnet-cli --model models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf
```

## Usage

### Low-Level API

```go
package main

import (
    "fmt"
    bitnet "github.com/hodizoda/go-bitnet-cli"
)

func main() {
    bitnet.Init()
    defer bitnet.Free()

    model, _ := bitnet.LoadModel("models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf")
    defer model.Close()

    ctx, _ := model.NewContext(
        bitnet.WithContextSize(2048),
        bitnet.WithTemperature(0.7),
    )
    defer ctx.Close()

    output, _ := ctx.Complete("The capital of Sweden is", 64)
    fmt.Println(output)
}
```

### LangChainGo Provider

```go
package main

import (
    "context"
    "fmt"
    bitnet "github.com/hodizoda/go-bitnet-cli"
    "github.com/tmc/langchaingo/llms"
)

func main() {
    llm, _ := bitnet.New("models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf",
        bitnet.WithMaxTokens(128),
        bitnet.WithLLMTemperature(0.7),
    )
    defer llm.Close()

    // Simple call
    resp, _ := llm.Call(context.Background(), "Explain quantum computing briefly.")
    fmt.Println(resp)

    // Streaming
    llm.Call(context.Background(), "Count to ten.",
        llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
            fmt.Print(string(chunk))
            return nil
        }),
    )
}
```

### CLI

```bash
# Interactive chat
go run ./cmd/bitnet-cli --model path/to/model.gguf

# Benchmark
go run ./cmd/bitnet-cli --mode bench --model path/to/model.gguf --prompt "Explain gravity." --max-tokens 256
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf` | Path to GGUF model |
| `--mode` | `chat` | `chat` or `bench` |
| `--prompt` | `Explain quantum computing in one paragraph.` | Prompt for bench mode |
| `--max-tokens` | `128` | Maximum tokens to generate |
| `--temp` | `0.7` | Sampling temperature |

## Build Targets

```bash
make libs           # Build bitnet.cpp static libraries
make models         # Download default model (bitnet-b1.58-2B-4T)
make models/falcon3-10b  # Download Falcon3-10B
make models/falcon3-7b   # Download Falcon3-7B
make models/falcon3-3b   # Download Falcon3-3B
make models/llama3-8b    # Download Llama3-8B
make test           # Build + download model + run tests
make bench MODEL=path/to/model.gguf  # Benchmark a model
make clean          # Remove build artifacts
```

## Architecture

```
bitnet.go       CGO bindings wrapping llama.h from bitnet.cpp
options.go      Functional options for model/context/LLM configuration
langchain.go    LangChainGo llms.Model adapter
cmd/bitnet-cli/ Interactive chat + benchmark CLI
3rdparty/BitNet git submodule (microsoft/BitNet fork of llama.cpp)
build/          cmake output (gitignored)
models/         GGUF model files (gitignored)
patches/        Const-correctness patch for bitnet.cpp
```

## Thread Safety

- `Model` is safe to share across goroutines
- `Context` is single-goroutine only (holds KV cache state)
- Create multiple contexts from one model for parallel inference

## Benchmarks

Measured on HP ProDesk 400 (i5-9500T, 24 GB RAM, Ubuntu 24.04):

| Model | Tokens/sec |
|-------|-----------|
| Falcon3-10B-Instruct | ~5.4 tok/s |

## License

MIT
