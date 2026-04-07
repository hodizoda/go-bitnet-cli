package bitnet

import (
	"context"
	"fmt"
	"strings"

	"github.com/tmc/langchaingo/llms"
)

// LLM implements the langchaingo llms.Model interface using BitNet inference.
type LLM struct {
	model *Model
	bCtx  *Context
	cfg   llmConfig
}

// Compile-time check that LLM implements llms.Model.
var _ llms.Model = (*LLM)(nil)

// New creates a new LLM from a GGUF model file.
func New(modelPath string, opts ...Option) (*LLM, error) {
	cfg := defaultLLMConfig()
	for _, o := range opts {
		o(&cfg)
	}

	model, err := LoadModel(modelPath)
	if err != nil {
		return nil, err
	}

	bCtx, err := model.NewContext(
		WithContextSize(cfg.contextSize),
		WithBatchSize(cfg.batchSize),
		WithTemperature(cfg.temperature),
		WithTopP(cfg.topP),
	)
	if err != nil {
		model.Close()
		return nil, err
	}

	return &LLM{model: model, bCtx: bCtx, cfg: cfg}, nil
}

// Close releases the model and context resources.
func (l *LLM) Close() error {
	l.bCtx.Close()
	l.model.Close()
	return nil
}

// Call implements the simplified text-in/text-out interface.
func (l *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	msg := llms.MessageContent{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.TextContent{Text: prompt}},
	}
	resp, err := l.GenerateContent(ctx, []llms.MessageContent{msg}, options...)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("bitnet: empty response")
	}
	return resp.Choices[0].Content, nil
}

// GenerateContent implements the llms.Model interface.
func (l *LLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	opts := llms.CallOptions{
		MaxTokens: l.cfg.maxTokens,
	}
	for _, o := range options {
		o(&opts)
	}

	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = l.cfg.maxTokens
	}

	prompt := l.formatMessages(messages)

	var streamFn func(token string) error
	if opts.StreamingFunc != nil {
		sf := opts.StreamingFunc
		streamFn = func(token string) error {
			return sf(ctx, []byte(token))
		}
	}

	result, err := l.bCtx.CompleteStreaming(ctx, prompt, maxTokens, streamFn)
	if err != nil {
		return nil, err
	}

	return &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				Content:    result,
				StopReason: "stop",
			},
		},
	}, nil
}

// formatMessages converts messages to a ChatML-formatted prompt string.
func (l *LLM) formatMessages(messages []llms.MessageContent) string {
	if l.cfg.chatTemplateFn != nil {
		return l.cfg.chatTemplateFn(messages)
	}
	return defaultChatMLTemplate(messages)
}

// defaultChatMLTemplate formats messages using the ChatML template.
func defaultChatMLTemplate(messages []llms.MessageContent) string {
	var sb strings.Builder
	for _, msg := range messages {
		role := chatMLRole(msg.Role)
		sb.WriteString("<|im_start|>")
		sb.WriteString(role)
		sb.WriteByte('\n')
		for _, part := range msg.Parts {
			if tc, ok := part.(llms.TextContent); ok {
				sb.WriteString(tc.Text)
			}
		}
		sb.WriteString("<|im_end|>\n")
	}
	sb.WriteString("<|im_start|>assistant\n")
	return sb.String()
}

func chatMLRole(role llms.ChatMessageType) string {
	switch role {
	case llms.ChatMessageTypeSystem:
		return "system"
	case llms.ChatMessageTypeHuman:
		return "user"
	case llms.ChatMessageTypeAI:
		return "assistant"
	default:
		return "user"
	}
}
