CC       ?= cc
CXX      ?= c++
PYTHON   ?= python3

BUILD_DIR := build
INC_FLAGS := -Iinclude -isystem tests

WARN_FLAGS := -Wall -Wextra -Wpedantic -Werror
C_STD      := -std=c99
CXX_STD    := -std=c++17

CFLAGS_BASE   := $(C_STD) $(WARN_FLAGS) $(INC_FLAGS)
CXXFLAGS_BASE := $(CXX_STD) $(WARN_FLAGS) $(INC_FLAGS)
LDFLAGS       := -lm

OPT_FLAGS := -O3
SAN_FLAGS := -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer

C_TESTS := $(patsubst tests/%.c,%,$(filter-out tests/test_amalgamation.c tests/test_amalgamation_impl.c,$(wildcard tests/test_*.c)))
CXX_TESTS := $(patsubst tests/%.cpp,%,$(wildcard tests/test_*.cpp))

.PHONY: all test test-simd test-scalar test-san test-cpp amalgamate test-amalgamation bench clean

all: test test-scalar test-san

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Native SIMD C99 tests
test: test-simd

test-simd: $(BUILD_DIR)
	@for t in $(C_TESTS); do \
		echo "[CC SIMD] $$t"; \
		$(CC) $(CFLAGS_BASE) $(OPT_FLAGS) tests/$$t.c -o $(BUILD_DIR)/$${t}_simd $(LDFLAGS) || exit 1; \
		./$(BUILD_DIR)/$${t}_simd || exit 1; \
	done

# Forced scalar fallback C99 tests
test-scalar: $(BUILD_DIR)
	@for t in $(C_TESTS); do \
		echo "[CC SCALAR] $$t"; \
		$(CC) $(CFLAGS_BASE) $(OPT_FLAGS) -DLA_FORCE_SCALAR=1 tests/$$t.c -o $(BUILD_DIR)/$${t}_scalar $(LDFLAGS) || exit 1; \
		./$(BUILD_DIR)/$${t}_scalar || exit 1; \
	done

# ASan + UBSan on both SIMD and scalar paths
test-san: $(BUILD_DIR)
	@for t in $(C_TESTS); do \
		echo "[CC SAN SIMD] $$t"; \
		$(CC) $(CFLAGS_BASE) $(SAN_FLAGS) tests/$$t.c -o $(BUILD_DIR)/$${t}_san_simd $(LDFLAGS) || exit 1; \
		./$(BUILD_DIR)/$${t}_san_simd || exit 1; \
		echo "[CC SAN SCALAR] $$t"; \
		$(CC) $(CFLAGS_BASE) $(SAN_FLAGS) -DLA_FORCE_SCALAR=1 tests/$$t.c -o $(BUILD_DIR)/$${t}_san_scalar $(LDFLAGS) || exit 1; \
		./$(BUILD_DIR)/$${t}_san_scalar || exit 1; \
	done

# C++17 tests (runs when tests/test_*.cpp exist)
test-cpp: $(BUILD_DIR)
	@for t in $(CXX_TESTS); do \
		echo "[CXX SIMD] $$t"; \
		$(CXX) $(CXXFLAGS_BASE) $(OPT_FLAGS) tests/$$t.cpp -o $(BUILD_DIR)/$${t}_cxx_simd $(LDFLAGS) || exit 1; \
		./$(BUILD_DIR)/$${t}_cxx_simd || exit 1; \
		echo "[CXX SAN] $$t"; \
		$(CXX) $(CXXFLAGS_BASE) $(SAN_FLAGS) tests/$$t.cpp -o $(BUILD_DIR)/$${t}_cxx_san $(LDFLAGS) || exit 1; \
		./$(BUILD_DIR)/$${t}_cxx_san || exit 1; \
	done

amalgamate:
	$(PYTHON) scripts/amalgamate.py

test-amalgamation: amalgamate $(BUILD_DIR)
	@echo "[AMALGAMATION C99 header-only]"
	$(CC) $(C_STD) $(WARN_FLAGS) -I. -isystem tests $(OPT_FLAGS) tests/test_amalgamation.c -o $(BUILD_DIR)/test_amalg_c $(LDFLAGS)
	./$(BUILD_DIR)/test_amalg_c
	@echo "[AMALGAMATION C99 multi-TU separate impl]"
	$(CC) $(C_STD) $(WARN_FLAGS) -I. -isystem tests $(OPT_FLAGS) -DLA_SEPARATE_IMPL -c tests/test_amalgamation.c -o $(BUILD_DIR)/amalg_main.o
	$(CC) $(C_STD) $(WARN_FLAGS) -I. -isystem tests $(OPT_FLAGS) -DLA_SEPARATE_IMPL -DLINALG_IMPLEMENTATION -c tests/test_amalgamation_impl.c -o $(BUILD_DIR)/amalg_impl.o
	$(CC) $(BUILD_DIR)/amalg_main.o $(BUILD_DIR)/amalg_impl.o -o $(BUILD_DIR)/test_amalg_multitu $(LDFLAGS)
	./$(BUILD_DIR)/test_amalg_multitu
	@echo "[AMALGAMATION C++17]"
	$(CXX) $(CXX_STD) $(WARN_FLAGS) -I. -isystem tests $(OPT_FLAGS) tests/test_cpp.cpp -DLA_USE_AMALGAMATION=1 -o $(BUILD_DIR)/test_amalg_cpp $(LDFLAGS)
	./$(BUILD_DIR)/test_amalg_cpp

bench: $(BUILD_DIR)
	$(CC) $(CFLAGS_BASE) $(OPT_FLAGS) bench/bench_linalg.c -o $(BUILD_DIR)/bench_simd $(LDFLAGS)
	$(CC) $(CFLAGS_BASE) $(OPT_FLAGS) -DLA_FORCE_SCALAR=1 bench/bench_linalg.c -o $(BUILD_DIR)/bench_scalar $(LDFLAGS)
	@echo "=== Native SIMD Benchmark ==="
	./$(BUILD_DIR)/bench_simd
	@echo "=== Scalar Fallback Benchmark ==="
	./$(BUILD_DIR)/bench_scalar

clean:
	rm -rf $(BUILD_DIR)
