package bitnet

/*
#cgo CFLAGS: -I${SRCDIR}/3rdparty/BitNet/3rdparty/llama.cpp/include -I${SRCDIR}/3rdparty/BitNet/3rdparty/llama.cpp/ggml/include -I${SRCDIR}/3rdparty/BitNet/include
#cgo LDFLAGS: -L${SRCDIR}/build/3rdparty/llama.cpp/src -L${SRCDIR}/build/3rdparty/llama.cpp/ggml/src -lllama -lggml -lm -lstdc++ -lpthread
#include <llama.h>
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"log"
	"runtime"
	"strings"
	"sync"
	"unsafe"
)

var initOnce sync.Once

// Init initializes the llama backend. Safe to call multiple times.
func Init() {
	initOnce.Do(func() {
		C.llama_backend_init()
	})
}

// Free releases the llama backend resources.
func Free() {
	C.llama_backend_free()
}

// Model wraps a loaded llama model.
type Model struct {
	model  *C.struct_llama_model
	closed bool
}

// Context wraps a llama inference context with a sampler chain.
type Context struct {
	ctx     *C.struct_llama_context
	model   *Model
	sampler *C.struct_llama_sampler
	closed  bool
}

// LoadModel loads a GGUF model file and returns a Model.
func LoadModel(path string, opts ...ModelOption) (*Model, error) {
	Init()

	cfg := defaultModelConfig()
	for _, o := range opts {
		o(&cfg)
	}

	params := C.llama_model_default_params()
	params.n_gpu_layers = C.int32_t(cfg.nGPULayers)
	params.use_mmap = C.bool(cfg.useMmap)

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cModel := C.llama_load_model_from_file(cPath, params)
	if cModel == nil {
		return nil, fmt.Errorf("bitnet: failed to load model from %q", path)
	}

	m := &Model{model: cModel}
	runtime.SetFinalizer(m, func(m *Model) {
		if !m.closed {
			log.Println("bitnet: Model was not closed before GC — call Close() explicitly")
		}
	})
	return m, nil
}

// Close frees the model. Idempotent.
func (m *Model) Close() {
	if m.closed {
		return
	}
	m.closed = true
	C.llama_free_model(m.model)
	runtime.SetFinalizer(m, nil)
}

// NewContext creates an inference context from the model.
func (m *Model) NewContext(opts ...ContextOption) (*Context, error) {
	cfg := defaultContextConfig()
	for _, o := range opts {
		o(&cfg)
	}

	params := C.llama_context_default_params()
	params.n_ctx = C.uint32_t(cfg.contextSize)
	params.n_batch = C.uint32_t(cfg.batchSize)
	params.no_perf = C.bool(true)

	cCtx := C.llama_new_context_with_model(m.model, params)
	if cCtx == nil {
		return nil, fmt.Errorf("bitnet: failed to create context")
	}

	// Build sampler chain: temp -> top_p -> dist
	chainParams := C.llama_sampler_chain_default_params()
	sampler := C.llama_sampler_chain_init(chainParams)
	C.llama_sampler_chain_add(sampler, C.llama_sampler_init_temp(C.float(cfg.temperature)))
	C.llama_sampler_chain_add(sampler, C.llama_sampler_init_top_p(C.float(cfg.topP), C.size_t(1)))
	C.llama_sampler_chain_add(sampler, C.llama_sampler_init_dist(C.uint32_t(cfg.seed)))

	ctx := &Context{ctx: cCtx, model: m, sampler: sampler}
	runtime.SetFinalizer(ctx, func(c *Context) {
		if !c.closed {
			log.Println("bitnet: Context was not closed before GC — call Close() explicitly")
		}
	})
	return ctx, nil
}

// Close frees the context and sampler. Idempotent.
func (c *Context) Close() {
	if c.closed {
		return
	}
	c.closed = true
	C.llama_sampler_free(c.sampler)
	C.llama_free(c.ctx)
	runtime.SetFinalizer(c, nil)
}

// Tokenize converts text to token IDs.
func (c *Context) Tokenize(text string) ([]int32, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	textLen := C.int32_t(len(text))

	// First call: get required buffer size (returns negative if buffer too small)
	n := C.llama_tokenize(c.model.model, cText, textLen, nil, 0, C.bool(true), C.bool(true))
	bufSize := n
	if bufSize < 0 {
		bufSize = -bufSize
	}
	if bufSize == 0 {
		return nil, nil
	}

	tokens := make([]C.llama_token, bufSize)
	n = C.llama_tokenize(c.model.model, cText, textLen, &tokens[0], bufSize, C.bool(true), C.bool(true))
	if n < 0 {
		return nil, fmt.Errorf("bitnet: tokenize failed, needed %d tokens", -n)
	}

	result := make([]int32, n)
	for i := C.int32_t(0); i < n; i++ {
		result[i] = int32(tokens[i])
	}
	return result, nil
}

// Detokenize converts token IDs back to text.
func (c *Context) Detokenize(tokens []int32) string {
	var sb strings.Builder
	buf := make([]C.char, 256)

	for _, tok := range tokens {
		n := C.llama_token_to_piece(c.model.model, C.llama_token(tok), &buf[0], C.int32_t(len(buf)), 0, C.bool(true))
		if n > 0 {
			sb.WriteString(C.GoStringN(&buf[0], n))
		} else if n < 0 {
			// Buffer too small, retry with larger buffer
			bigBuf := make([]C.char, -n)
			n2 := C.llama_token_to_piece(c.model.model, C.llama_token(tok), &bigBuf[0], C.int32_t(-n), 0, C.bool(true))
			if n2 > 0 {
				sb.WriteString(C.GoStringN(&bigBuf[0], n2))
			}
		}
	}
	return sb.String()
}

// Decode processes a batch of tokens through the model.
func (c *Context) Decode(tokens []int32) error {
	n := len(tokens)
	if n == 0 {
		return nil
	}

	batch := C.llama_batch_init(C.int32_t(n), 0, 1)
	defer C.llama_batch_free(batch)

	batch.n_tokens = C.int32_t(n)

	batchTokens := unsafe.Slice(batch.token, n)
	batchPos := unsafe.Slice(batch.pos, n)
	batchNSeqID := unsafe.Slice(batch.n_seq_id, n)
	batchSeqID := unsafe.Slice(batch.seq_id, n)
	batchLogits := unsafe.Slice(batch.logits, n)

	for i := 0; i < n; i++ {
		batchTokens[i] = C.llama_token(tokens[i])
		batchPos[i] = C.llama_pos(i)
		batchNSeqID[i] = 1
		*batchSeqID[i] = 0
		if i == n-1 {
			batchLogits[i] = 1 // only compute logits for last token
		} else {
			batchLogits[i] = 0
		}
	}

	rc := C.llama_decode(c.ctx, batch)
	if rc != 0 {
		return fmt.Errorf("bitnet: decode failed with code %d", rc)
	}
	return nil
}

// Sample picks the next token from the model's output distribution.
func (c *Context) Sample() (int32, error) {
	tok := C.llama_sampler_sample(c.sampler, c.ctx, C.int32_t(-1))
	return int32(tok), nil
}

// decodeSingle decodes a single token at the given position.
func (c *Context) decodeSingle(token int32, pos int) error {
	batch := C.llama_batch_init(1, 0, 1)
	defer C.llama_batch_free(batch)

	batch.n_tokens = 1

	batchTokens := unsafe.Slice(batch.token, 1)
	batchPos := unsafe.Slice(batch.pos, 1)
	batchNSeqID := unsafe.Slice(batch.n_seq_id, 1)
	batchSeqID := unsafe.Slice(batch.seq_id, 1)
	batchLogits := unsafe.Slice(batch.logits, 1)

	batchTokens[0] = C.llama_token(token)
	batchPos[0] = C.llama_pos(pos)
	batchNSeqID[0] = 1
	*batchSeqID[0] = 0
	batchLogits[0] = 1

	rc := C.llama_decode(c.ctx, batch)
	if rc != 0 {
		return fmt.Errorf("bitnet: decode single failed with code %d", rc)
	}
	return nil
}

// Complete runs auto-regressive generation given a prompt.
func (c *Context) Complete(prompt string, maxTokens int) (string, error) {
	tokens, err := c.Tokenize(prompt)
	if err != nil {
		return "", fmt.Errorf("bitnet: tokenize: %w", err)
	}
	if len(tokens) == 0 {
		return "", fmt.Errorf("bitnet: empty prompt after tokenization")
	}

	if err := c.Decode(tokens); err != nil {
		return "", fmt.Errorf("bitnet: decode prompt: %w", err)
	}

	eosToken := int32(C.llama_token_eos(c.model.model))

	var generated []int32
	for i := 0; i < maxTokens; i++ {
		tok, err := c.Sample()
		if err != nil {
			return "", fmt.Errorf("bitnet: sample: %w", err)
		}
		if tok == eosToken {
			break
		}
		generated = append(generated, tok)
		if err := c.decodeSingle(tok, len(tokens)+len(generated)-1); err != nil {
			return "", fmt.Errorf("bitnet: decode generated token: %w", err)
		}
	}

	return c.Detokenize(generated), nil
}
