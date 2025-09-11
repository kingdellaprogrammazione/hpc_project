# Compiler and flags
MCC = mpicc
CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -lm

# Directories
SRC_DIR = src/c
TEST_DIR = test
EXTERNAL_DIR = external/unity
BUILD_DIR = build
LOG_DIR = data/logs

# Sources used in app and tests
SHARED_SRCS = $(SRC_DIR)/mpi_functions.c $(SRC_DIR)/structs.c $(SRC_DIR)/io.c

# App-specific sources
APP_SRCS = $(SRC_DIR)/mpi_final.c $(SRC_DIR)/io.c $(SRC_DIR)/mpi_ops.c  $(SHARED_SRCS)

# Test sources
TEST_SRCS = $(TEST_DIR)/testfile.c

UNITY_SRC = $(EXTERNAL_DIR)/unity.c

# Object files
APP_OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(APP_SRCS))
TEST_OBJS = $(patsubst $(TEST_DIR)/%.c, $(BUILD_DIR)/%.o, $(TEST_SRCS))
SHARED_OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SHARED_SRCS))
UNITY_OBJ = $(BUILD_DIR)/unity.o

all: test gen run check 

# Build app executable
mpi_mm: $(APP_OBJS) $(SHARED_OBJS)
	$(MCC) $(CFLAGS) $^ -o $(BUILD_DIR)/mpi_mm.out $(LDFLAGS)

# Build test runner executable (includes shared code, tests, unity)
test.out: $(SHARED_OBJS) $(TEST_OBJS) $(UNITY_OBJ)
	$(MCC) $(CFLAGS) $^ -o $(BUILD_DIR)/test.out  $(LDFLAGS)

# Compile object files for src and test
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(MCC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(MCC) $(CFLAGS) -c $< -o $@

$(UNITY_OBJ): $(UNITY_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create build directory if missing
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

gen: 
	python3 ./src/python/matrix_generator.py 

check: 
	python3 ./src/python/matrix_checker.py

run: mpi_mm
	mpirun -np 7 ./build/mpi_mm.out

# Run tests
test: test.out
	./$(BUILD_DIR)/test.out

clean:
	rm -r $(BUILD_DIR)/* $(LOG_DIR)/log*

.PHONY: all test gen check clean mpi_mm
