CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic

.PHONY: all train train-gru chat chat-mlp test test-response-quality test-conversation-context test-conversation-manager clean

GRU_DIR = experiments/mikuru-v3/gru-v1

all: build/first-neuron build/mikuru-train build/mikuru-chat build/mikuru-chat-mlp build/mikuru-system-inspector

build:
	mkdir -p build

build/first-neuron: src/main.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

build/mikuru-train: src/train_dialogue.cpp src/dialogue_common.hpp | build
	$(CXX) $(CXXFLAGS) src/train_dialogue.cpp -o $@

build/mikuru-system-inspector: src/system_inspector.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

build/test-intent-router: src/test_intent_router.cpp src/intent_router.hpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

build/mikuru-chat-mlp: src/chat.cpp src/dialogue_common.hpp src/conversation_manager.hpp src/system_inspector.hpp src/intent_router.hpp src/system_personality.hpp src/retrieval.hpp src/response_quality.hpp data/retrieval_dialogue.tsv build/mikuru-system-inspector | build
	$(CXX) $(CXXFLAGS) src/chat.cpp -o $@

build/mikuru-chat: src/chat_gru.cpp src/conversation_context.hpp src/conversation_manager.hpp src/terminal_ui.hpp src/response_quality.hpp $(GRU_DIR)/src/gru_model.hpp src/dialogue_common.hpp src/system_inspector.hpp src/intent_router.hpp src/system_personality.hpp src/retrieval.hpp data/retrieval_dialogue.tsv build
	$(CXX) $(CXXFLAGS) src/chat_gru.cpp -o $@

build/mikuru-train-gru: $(GRU_DIR)/src/train_gru.cpp $(GRU_DIR)/src/gru_model.hpp src/dialogue_common.hpp | build
	$(CXX) $(CXXFLAGS) $(GRU_DIR)/src/train_gru.cpp -o $@

build/test-response-quality: src/test_response_quality.cpp src/response_quality.hpp src/dialogue_common.hpp | build
	$(CXX) $(CXXFLAGS) src/test_response_quality.cpp -o $@

build/test-conversation-context: src/test_conversation_context.cpp src/conversation_context.hpp $(GRU_DIR)/src/gru_model.hpp src/dialogue_common.hpp | build
	$(CXX) $(CXXFLAGS) src/test_conversation_context.cpp -o $@

build/test-conversation-manager: src/test_conversation_manager.cpp src/conversation_manager.hpp src/dialogue_common.hpp | build
	$(CXX) $(CXXFLAGS) src/test_conversation_manager.cpp -o $@

train: build/mikuru-train
	./build/mikuru-train

train-gru: build/mikuru-train-gru
	@if [ -e build/gru/gru-model.txt ] || [ -e build/gru/gru-vocab.txt ]; then \
		echo "Refusing to overwrite build/gru GRU artifacts."; exit 1; \
	fi
	./build/mikuru-train-gru --corpus $(GRU_DIR)/data/dialogue.tsv --out build/gru

chat: build/mikuru-chat
	@vocab="$(GRU_DIR)/build/gru-vocab.txt"; \
	model="$(GRU_DIR)/build/gru-model.txt"; \
	if [ -f build/gru/gru-vocab.txt ] && [ -f build/gru/gru-model.txt ]; then \
		vocab="build/gru/gru-vocab.txt"; model="build/gru/gru-model.txt"; \
	fi; \
	if [ ! -f "$$vocab" ] || [ ! -f "$$model" ]; then \
		echo "No matching GRU vocabulary/model pair found."; \
		echo "Use the preserved experiment artifacts or run 'make train-gru'."; \
		exit 1; \
	fi; \
	./build/mikuru-chat --vocab "$$vocab" --model "$$model"

chat-mlp: build/mikuru-chat-mlp
	@if [ ! -f build/mikuru-dialogue.weights ] || [ ! -f build/dialogue-vocab.txt ]; then \
		echo "No trained MLP model found. Run 'make train' first."; exit 1; \
	fi
	./build/mikuru-chat-mlp

test-response-quality: build/test-response-quality
	./build/test-response-quality

test-conversation-context: build/test-conversation-context
	./build/test-conversation-context

test-conversation-manager: build/test-conversation-manager
	./build/test-conversation-manager

test: all build/test-intent-router build/test-response-quality
	./build/test-intent-router
	./build/test-response-quality
	./build/first-neuron | grep -q "Mikuru's first neuron: 7"
	./build/mikuru-train > build/test-training.log
	grep -q "Saved model:" build/test-training.log
	grep -q "Prompt conditioning: mean of full user prompt" build/test-training.log
	grep -Eq "Parameters: [0-9]+" build/test-training.log
	printf '/help\nHello there!\nHow much disk space do I have?\nHow much RAM is available?\n/reset\n/quit\n' | ./build/mikuru-chat-mlp > build/test-chat.log
	grep -q "Mikuru v0.2 prompt-conditioned chat" build/test-chat.log
	printf 'Explain neural networks\\n/quit\\n' | ./build/mikuru-chat-mlp > build/test-chat-retrieval.log
	grep -q "A neural network combines simple calculations with adjustable parameters" build/test-chat-retrieval.log
	grep -q "Context cleared" build/test-chat.log
	grep -q "Quick check: your system storage" build/test-chat.log
	grep -q "Quick check: your RAM" build/test-chat.log
	printf '/help\nMy program crashed.\n/quit\n' | ./build/mikuru-chat-mlp > build/test-router-negative.log
	! grep -q "Quick check: your RAM" build/test-router-negative.log
	printf 'Show me disk space and memory.\n/quit\n' | ./build/mikuru-chat-mlp > build/test-router-combined.log
	grep -q "Quick check: your system storage" build/test-router-combined.log
	grep -q "Quick check: your RAM" build/test-router-combined.log
	! grep -E '<PAD>|<UNK>|<BOS>|<USER>|<ASSISTANT>' build/test-chat.log
	@echo "All smoke tests passed. See build/test-training.log and build/test-chat.log."

clean:
	rm -rf build
