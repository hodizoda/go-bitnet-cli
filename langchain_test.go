package bitnet

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

func TestLLMCall(t *testing.T) {
	skipIfNoModel(t)

	l, err := New(testModelPath, WithMaxTokens(32))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer l.Close()

	resp, err := l.Call(context.Background(), "What is 2+2?")
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if resp == "" {
		t.Fatal("expected non-empty response")
	}
	t.Logf("response: %q", resp)
}

func TestLLMGenerateContent(t *testing.T) {
	skipIfNoModel(t)

	l, err := New(testModelPath, WithMaxTokens(32))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer l.Close()

	messages := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeSystem,
			Parts: []llms.ContentPart{llms.TextContent{Text: "You are a helpful assistant."}},
		},
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextContent{Text: "Say hello."}},
		},
	}

	resp, err := l.GenerateContent(context.Background(), messages)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}
	if resp.Choices[0].Content == "" {
		t.Fatal("expected non-empty content")
	}
	t.Logf("response: %q", resp.Choices[0].Content)
}

func TestLLMStreaming(t *testing.T) {
	skipIfNoModel(t)

	l, err := New(testModelPath, WithMaxTokens(32))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer l.Close()

	var chunks int64
	resp, err := l.Call(context.Background(), "Count to five.",
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			atomic.AddInt64(&chunks, 1)
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("Call with streaming: %v", err)
	}
	if resp == "" {
		t.Fatal("expected non-empty response")
	}
	if chunks == 0 {
		t.Fatal("expected streaming chunks > 0")
	}
	t.Logf("response: %q (chunks: %d)", resp, chunks)
}

func TestLLMCancellation(t *testing.T) {
	skipIfNoModel(t)

	l, err := New(testModelPath, WithMaxTokens(256))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer l.Close()

	ctx, cancel := context.WithCancel(context.Background())

	var chunks int64
	_, err = l.Call(ctx, "Write a long story about a dragon.",
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			n := atomic.AddInt64(&chunks, 1)
			if n >= 3 {
				cancel()
			}
			return nil
		}),
	)

	got := atomic.LoadInt64(&chunks)
	if got < 3 {
		t.Fatalf("expected at least 3 chunks before cancel, got %d", got)
	}
	if err == nil {
		t.Log("generation completed before cancellation took effect (EOS reached early)")
	} else if err != context.Canceled {
		t.Fatalf("expected context.Canceled, got: %v", err)
	} else {
		t.Logf("cancelled after %d chunks", got)
	}
}
