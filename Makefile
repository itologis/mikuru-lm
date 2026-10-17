CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic

.PHONY: all train chat test clean

all: build/first-neuron build/mikuru-train build/mikuru-chat

build:
	mkdir -p build

build/first-neuron: src/main.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

build/mikuru-train: src/train_dialogue.cpp src/dialogue_common.hpp | build
	$(CXX) $(CXXFLAGS) src/train_dialogue.cpp -o $@

build/mikuru-chat: src/chat.cpp src/dialogue_common.hpp | build
	$(CXX) $(CXXFLAGS) src/chat.cpp -o $@

train: build/mikuru-train
	./build/mikuru-train

chat: build/mikuru-chat
	@if [ ! -f build/mikuru-dialogue.weights ] || [ ! -f build/dialogue-vocab.txt ]; then \
		echo "No trained dialogue model found. Run 'make train' first."; exit 1; \
	fi
	./build/mikuru-chat

test: all
	./build/first-neuron | grep -q "Mikuru's first neuron: 7"
	./build/mikuru-train > build/test-training.log
	grep -q "Saved model:" build/test-training.log
	printf '/help\nHello there!\n/reset\n/quit\n' | ./build/mikuru-chat > build/test-chat.log
	grep -q "Mikuru v0.1 experimental chat" build/test-chat.log
	grep -q "Context cleared" build/test-chat.log
	! grep -E '<PAD>|<UNK>|<BOS>|<USER>|<ASSISTANT>' build/test-chat.log
	@echo "All smoke tests passed. See build/test-training.log and build/test-chat.log."

clean:
	rm -rf build
