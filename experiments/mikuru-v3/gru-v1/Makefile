CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic

ROOT_HEADER = ../../../src/dialogue_common.hpp

.PHONY: all train chat test
all: build/gru-train build/gru-chat build/test-gru

build:
	mkdir -p build

build/gru-train: src/train_gru.cpp src/gru_model.hpp $(ROOT_HEADER) | build
	$(CXX) $(CXXFLAGS) src/train_gru.cpp -o $@

build/gru-chat: src/chat_gru.cpp src/gru_model.hpp $(ROOT_HEADER) | build
	$(CXX) $(CXXFLAGS) src/chat_gru.cpp -o $@

build/test-gru: tests/test_gru.cpp src/gru_model.hpp $(ROOT_HEADER) | build
	$(CXX) $(CXXFLAGS) tests/test_gru.cpp -o $@

train: build/gru-train
	./build/gru-train --corpus data/dialogue.tsv --out build

chat: build/gru-chat
	@if [ ! -f build/gru-vocab.txt ] || [ ! -f build/gru-model.txt ]; then \
		echo "No GRU model yet. Run 'make train' first."; exit 1; \
	fi
	./build/gru-chat

test: all
	./build/test-gru
	./build/gru-train --corpus data/dialogue.tsv --out build > build/test-training.log
	grep -q "Saved GRU model:" build/test-training.log
	printf 'Tell me about neural networks\n/quit\n' | ./build/gru-chat > build/test-chat.log
	grep -q "Mikuru GRU V1" build/test-chat.log
	! grep -E '<PAD>|<UNK>|<BOS>|<USER>|<ASSISTANT>' build/test-chat.log
	@echo "GRU tests passed. See build/test-training.log and build/test-chat.log."
