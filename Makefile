CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic

.PHONY: all train chat test clean

all: build/first-neuron build/mikuru-train build/mikuru-chat build/mikuru-system-inspector

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

build/mikuru-chat: src/chat.cpp src/dialogue_common.hpp src/system_inspector.hpp src/intent_router.hpp src/system_personality.hpp src/retrieval.hpp data/retrieval_dialogue.tsv build/mikuru-system-inspector | build
	$(CXX) $(CXXFLAGS) src/chat.cpp -o $@

train: build/mikuru-train
	./build/mikuru-train

chat: build/mikuru-chat
	@if [ ! -f build/mikuru-dialogue.weights ] || [ ! -f build/dialogue-vocab.txt ]; then \
		echo "No trained dialogue model found. Run 'make train' first."; exit 1; \
	fi
	./build/mikuru-chat

test: all build/test-intent-router
	./build/test-intent-router
	./build/first-neuron | grep -q "Mikuru's first neuron: 7"
	./build/mikuru-train > build/test-training.log
	grep -q "Saved model:" build/test-training.log
	grep -q "Prompt conditioning: mean of full user prompt" build/test-training.log
	grep -Eq "Parameters: [0-9]+" build/test-training.log
	printf '/help\nHello there!\nHow much disk space do I have?\nHow much RAM is available?\n/reset\n/quit\n' | ./build/mikuru-chat > build/test-chat.log
	grep -q "Mikuru v0.2 prompt-conditioned chat" build/test-chat.log
	printf 'Explain neural networks\\n/quit\\n' | ./build/mikuru-chat > build/test-chat-retrieval.log
	grep -q "A neural network combines simple calculations with adjustable parameters" build/test-chat-retrieval.log
	grep -q "Context cleared" build/test-chat.log
	grep -q "Quick check: your system storage" build/test-chat.log
	grep -q "Quick check: your RAM" build/test-chat.log
	printf '/help\nMy program crashed.\n/quit\n' | ./build/mikuru-chat > build/test-router-negative.log
	! grep -q "Quick check: your RAM" build/test-router-negative.log
	printf 'Show me disk space and memory.\n/quit\n' | ./build/mikuru-chat > build/test-router-combined.log
	grep -q "Quick check: your system storage" build/test-router-combined.log
	grep -q "Quick check: your RAM" build/test-router-combined.log
	! grep -E '<PAD>|<UNK>|<BOS>|<USER>|<ASSISTANT>' build/test-chat.log
	@echo "All smoke tests passed. See build/test-training.log and build/test-chat.log."

clean:
	rm -rf build
