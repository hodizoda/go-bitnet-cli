package bitnet

import "github.com/tmc/langchaingo/llms"

// modelConfig holds configuration for loading a model.
type modelConfig struct {
	nGPULayers int
	useMmap    bool
}

func defaultModelConfig() modelConfig {
	return modelConfig{
		nGPULayers: 0,
		useMmap:    true,
	}
}

// ModelOption configures model loading.
type ModelOption func(*modelConfig)

// WithNGPULayers sets the number of layers to offload to GPU.
func WithNGPULayers(n int) ModelOption {
	return func(c *modelConfig) { c.nGPULayers = n }
}

// WithUseMmap sets whether to use memory-mapped file I/O for loading.
func WithUseMmap(v bool) ModelOption {
	return func(c *modelConfig) { c.useMmap = v }
}

// contextConfig holds configuration for creating a context.
type contextConfig struct {
	contextSize uint32
	batchSize   uint32
	temperature float32
	topP        float32
	seed        uint32
}

func defaultContextConfig() contextConfig {
	return contextConfig{
		contextSize: 0, // 0 = use model's native training context size
		batchSize:   512,
		temperature: 0.7,
		topP:        0.9,
		seed:        0,
	}
}

// ContextOption configures context creation.
type ContextOption func(*contextConfig)

// WithContextSize sets the context window size (number of tokens).
func WithContextSize(n uint32) ContextOption {
	return func(c *contextConfig) { c.contextSize = n }
}

// WithBatchSize sets the maximum batch size for prompt processing.
func WithBatchSize(n uint32) ContextOption {
	return func(c *contextConfig) { c.batchSize = n }
}

// WithTemperature sets the sampling temperature.
func WithTemperature(t float32) ContextOption {
	return func(c *contextConfig) { c.temperature = t }
}

// WithTopP sets the top-p (nucleus) sampling threshold.
func WithTopP(p float32) ContextOption {
	return func(c *contextConfig) { c.topP = p }
}

// WithSeed sets the random seed for sampling.
func WithSeed(s uint32) ContextOption {
	return func(c *contextConfig) { c.seed = s }
}

// llmConfig holds configuration for the LangChainGo LLM adapter.
type llmConfig struct {
	contextSize    uint32
	batchSize      uint32
	temperature    float32
	topP           float32
	maxTokens      int
	chatTemplateFn func(messages []llms.MessageContent) string
}

func defaultLLMConfig() llmConfig {
	return llmConfig{
		contextSize: 0, // 0 = use model's native training context size
		batchSize:   512,
		temperature: 0.7,
		topP:        0.9,
		maxTokens:   256,
	}
}

// Option configures the LangChainGo LLM adapter.
type Option func(*llmConfig)

// WithLLMTemperature sets the sampling temperature for the LLM adapter.
func WithLLMTemperature(t float32) Option {
	return func(c *llmConfig) { c.temperature = t }
}

// WithLLMTopP sets the top-p sampling threshold for the LLM adapter.
func WithLLMTopP(p float32) Option {
	return func(c *llmConfig) { c.topP = p }
}

// WithLLMContextSize sets the context window size for the LLM adapter.
func WithLLMContextSize(n uint32) Option {
	return func(c *llmConfig) { c.contextSize = n }
}

// WithMaxTokens sets the default maximum tokens for generation.
func WithMaxTokens(n int) Option {
	return func(c *llmConfig) { c.maxTokens = n }
}

// WithChatTemplate sets a custom chat template function.
func WithChatTemplate(fn func([]llms.MessageContent) string) Option {
	return func(c *llmConfig) { c.chatTemplateFn = fn }
}
