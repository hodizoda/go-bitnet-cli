package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	bitnet "go-bitnet"

	"github.com/tmc/langchaingo/llms"
)

func main() {
	modelPath := flag.String("model", "models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf", "path to GGUF model")
	mode := flag.String("mode", "chat", "mode: chat or bench")
	prompt := flag.String("prompt", "Explain quantum computing in one paragraph.", "prompt for bench mode")
	maxTokens := flag.Int("max-tokens", 128, "maximum tokens to generate")
	ctxSize := flag.Int("ctx-size", 0, "context window size in tokens (0 = model's native size)")
	temp := flag.Float64("temp", 0.7, "sampling temperature")
	flag.Parse()

	switch *mode {
	case "chat":
		runChat(*modelPath, *maxTokens, uint32(*ctxSize), float32(*temp))
	case "bench":
		runBench(*modelPath, *prompt, *maxTokens, uint32(*ctxSize), float32(*temp))
	default:
		fmt.Fprintf(os.Stderr, "unknown mode: %s (use chat or bench)\n", *mode)
		os.Exit(1)
	}
}

func runChat(modelPath string, maxTokens int, ctxSize uint32, temp float32) {
	llm, err := bitnet.New(modelPath,
		bitnet.WithMaxTokens(maxTokens),
		bitnet.WithLLMContextSize(ctxSize),
		bitnet.WithLLMTemperature(temp),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading model: %v\n", err)
		os.Exit(1)
	}
	defer llm.Close()

	fmt.Println("BitNet chat (type 'quit' to exit)")
	scanner := bufio.NewScanner(os.Stdin)

	var history []llms.MessageContent

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if line == "quit" || line == "exit" {
			break
		}

		// Add user message to history
		history = append(history, llms.MessageContent{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextContent{Text: line}},
		})

		// Clear KV cache so the full conversation is re-processed
		llm.Reset()

		resp, err := llm.GenerateContent(context.Background(), history,
			llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
				fmt.Print(string(chunk))
				return nil
			}),
		)
		fmt.Println()

		contextFull := err != nil && strings.Contains(err.Error(), "context window full")

		if err != nil && !contextFull {
			// Prompt itself too large — trim oldest messages and retry
			if strings.Contains(err.Error(), "exceeds context size") {
				fmt.Fprintf(os.Stderr, "(context full, trimming old messages)\n")
				start := 0
				if len(history) > 0 && history[0].Role == llms.ChatMessageTypeSystem {
					start = 1
				}
				// Drop oldest pair (user+assistant)
				if len(history)-start > 2 {
					history = append(history[:start], history[start+2:]...)
				}
			} else {
				fmt.Fprintf(os.Stderr, "error: %v\n", err)
			}
			continue
		}

		if contextFull {
			fmt.Fprintf(os.Stderr, "\n(context window full — response was cut short, trimming old messages)\n")
			start := 0
			if len(history) > 0 && history[0].Role == llms.ChatMessageTypeSystem {
				start = 1
			}
			if len(history)-start > 2 {
				history = append(history[:start], history[start+2:]...)
			}
		}

		// Add assistant response to history (even if truncated by context limit)
		content := ""
		if resp != nil && len(resp.Choices) > 0 {
			content = resp.Choices[0].Content
		}
		if content != "" {
			history = append(history, llms.MessageContent{
				Role:  llms.ChatMessageTypeAI,
				Parts: []llms.ContentPart{llms.TextContent{Text: content}},
			})
		}
	}
}

func runBench(modelPath, prompt string, maxTokens int, ctxSize uint32, temp float32) {
	fmt.Printf("Loading model: %s\n", modelPath)
	loadStart := time.Now()

	llm, err := bitnet.New(modelPath,
		bitnet.WithMaxTokens(maxTokens),
		bitnet.WithLLMContextSize(ctxSize),
		bitnet.WithLLMTemperature(temp),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading model: %v\n", err)
		os.Exit(1)
	}
	defer llm.Close()
	fmt.Printf("Model loaded in %v\n", time.Since(loadStart))

	fmt.Printf("Prompt: %q\n", prompt)
	fmt.Printf("Max tokens: %d\n\n", maxTokens)

	var (
		tokenCount int
		firstToken time.Duration
		genStart   time.Time
	)

	genStart = time.Now()
	result, err := llm.Call(context.Background(), prompt,
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			tokenCount++
			if tokenCount == 1 {
				firstToken = time.Since(genStart)
			}
			return nil
		}),
	)

	elapsed := time.Since(genStart)

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)

	fmt.Printf("Output: %s\n\n", result)
	fmt.Println("--- Benchmark Results ---")
	fmt.Printf("Tokens:          %d\n", tokenCount)
	fmt.Printf("Total time:      %v\n", elapsed)
	fmt.Printf("Time to first:   %v\n", firstToken)
	if tokenCount > 0 {
		tps := float64(tokenCount) / elapsed.Seconds()
		fmt.Printf("Tokens/sec:      %.2f\n", tps)
	}
	fmt.Printf("Go heap:         %.1f MB\n", float64(mem.HeapAlloc)/1024/1024)
}
