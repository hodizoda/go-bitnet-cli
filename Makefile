.PHONY: libs clean test bench models patch

BUILD_DIR := $(CURDIR)/build
BITNET_DIR := $(CURDIR)/3rdparty/BitNet
KERNELS_SRC := $(BITNET_DIR)/preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl2.h
KERNELS_DST := $(BITNET_DIR)/include/bitnet-lut-kernels.h
PATCH_FILE := $(CURDIR)/patches/0001-fix-const-correctness-ggml-bitnet-mad.patch

libs: patch $(KERNELS_DST)
	cmake -B $(BUILD_DIR) -S $(BITNET_DIR) \
		-DBITNET_X86_TL2=OFF \
		-DBUILD_SHARED_LIBS=OFF \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_SERVER=OFF
	cmake --build $(BUILD_DIR) --config Release -j$$(nproc)

patch:
	cd $(BITNET_DIR) && git apply --check $(PATCH_FILE) 2>/dev/null && git apply $(PATCH_FILE) || true

$(KERNELS_DST): $(KERNELS_SRC)
	cp $< $@

-include .env
export HF_TOKEN

DL = python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('$(1)', '$(2)', local_dir='$(3)')"

models: models/bitnet-b1.58-2B-4T

models/bitnet-b1.58-2B-4T:
	$(call DL,microsoft/bitnet-b1.58-2B-4T-gguf,ggml-model-i2_s.gguf,$@)

models/falcon3-10b:
	$(call DL,tiiuae/Falcon3-10B-Instruct-1.58bit-GGUF,ggml-model-i2_s.gguf,$@)

models/falcon3-7b:
	$(call DL,tiiuae/Falcon3-7B-Instruct-1.58bit-GGUF,ggml-model-i2_s.gguf,$@)

models/falcon3-3b:
	$(call DL,tiiuae/Falcon3-3B-Instruct-1.58bit-GGUF,ggml-model-i2_s.gguf,$@)

models/llama3-8b:
	$(call DL,huyrua1996/Llama3-8B-1.58-100B-tokens-bitnet,ggml-model-i2_s.gguf,$@)

test: libs
	go test -v -count=1 ./...

bench: libs
	go run ./cmd/bitnet-cli --mode bench --model $(MODEL)

clean:
	rm -rf $(BUILD_DIR)
