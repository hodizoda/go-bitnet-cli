package bitnet

import (
	"os"
	"testing"
)

const testModelPath = "models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf"

func skipIfNoModel(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(testModelPath); os.IsNotExist(err) {
		t.Skip("model not downloaded — run: make models")
	}
}

func TestInitFree(t *testing.T) {
	Init()
	Free()
}

func TestLoadModel(t *testing.T) {
	skipIfNoModel(t)
	Init()

	m, err := LoadModel(testModelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	if m.model == nil {
		t.Fatal("model pointer is nil")
	}
}

func TestNewContext(t *testing.T) {
	skipIfNoModel(t)
	Init()

	m, err := LoadModel(testModelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	ctx, err := m.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	if ctx.ctx == nil {
		t.Fatal("context pointer is nil")
	}
}

func TestTokenize(t *testing.T) {
	skipIfNoModel(t)
	Init()

	m, err := LoadModel(testModelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	ctx, err := m.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	tokens, err := ctx.Tokenize("Hello, world!")
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("expected non-empty token list")
	}
	t.Logf("tokens: %v (len=%d)", tokens, len(tokens))
}

func TestDetokenize(t *testing.T) {
	skipIfNoModel(t)
	Init()

	m, err := LoadModel(testModelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	ctx, err := m.NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	tokens, err := ctx.Tokenize("Hello")
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}

	text := ctx.Detokenize(tokens)
	if text == "" {
		t.Fatal("expected non-empty detokenized text")
	}
	t.Logf("detokenized: %q", text)
}

func TestDecodeSample(t *testing.T) {
	skipIfNoModel(t)
	Init()

	m, err := LoadModel(testModelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	ctx, err := m.NewContext(WithTemperature(0.0))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	tokens, err := ctx.Tokenize("The capital of France is")
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}

	if err := ctx.Decode(tokens); err != nil {
		t.Fatalf("Decode: %v", err)
	}

	tok, err := ctx.Sample()
	if err != nil {
		t.Fatalf("Sample: %v", err)
	}

	piece := ctx.Detokenize([]int32{tok})
	if piece == "" {
		t.Fatal("expected non-empty sampled piece")
	}
	t.Logf("sampled token %d -> %q", tok, piece)
}

func TestComplete(t *testing.T) {
	skipIfNoModel(t)
	Init()

	m, err := LoadModel(testModelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	ctx, err := m.NewContext(WithTemperature(0.0))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	out, err := ctx.Complete("The capital of Sweden is", 32)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if out == "" {
		t.Fatal("expected non-empty completion")
	}
	t.Logf("completion: %q", out)
}

func TestCompleteWithOptions(t *testing.T) {
	skipIfNoModel(t)
	Init()

	m, err := LoadModel(testModelPath, WithUseMmap(false))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	ctx, err := m.NewContext(
		WithContextSize(1024),
		WithBatchSize(256),
		WithTemperature(0.5),
		WithTopP(0.8),
	)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	out, err := ctx.Complete("Hello", 16)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if out == "" {
		t.Fatal("expected non-empty completion")
	}
	t.Logf("completion with options: %q", out)
}
