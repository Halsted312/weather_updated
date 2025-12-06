# ============================================================================
# Makefile for Enhanced Marketing Agent
# ============================================================================

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic
DEBUGFLAGS = -g -DDEBUG
RELEASEFLAGS = -O3 -DNDEBUG

# Directories
SRC_DIR = .
BUILD_DIR = build
BIN_DIR = bin

# Files
LIBRARY_SRC = enhanced_marketing_agent.cpp
LIBRARY_OBJ = $(BUILD_DIR)/enhanced_marketing_agent.o
LIBRARY = $(BUILD_DIR)/libmarketing_agent.a

DEMO_SRC = demo.cpp
DEMO_OBJ = $(BUILD_DIR)/demo.o
DEMO_BIN = $(BIN_DIR)/marketing_demo

# Default target
.PHONY: all
all: release

# Release build
.PHONY: release
release: CXXFLAGS += $(RELEASEFLAGS)
release: directories $(DEMO_BIN)
	@echo ""
	@echo "═══════════════════════════════════════════════════════"
	@echo "  Build complete! (Release mode)"
	@echo "═══════════════════════════════════════════════════════"
	@echo "  Executable: $(DEMO_BIN)"
	@echo "  To run: make run"
	@echo "═══════════════════════════════════════════════════════"
	@echo ""

# Debug build
.PHONY: debug
debug: CXXFLAGS += $(DEBUGFLAGS)
debug: directories $(DEMO_BIN)
	@echo ""
	@echo "═══════════════════════════════════════════════════════"
	@echo "  Build complete! (Debug mode)"
	@echo "═══════════════════════════════════════════════════════"
	@echo "  Executable: $(DEMO_BIN)"
	@echo "  To run: make run"
	@echo "═══════════════════════════════════════════════════════"
	@echo ""

# Create necessary directories
.PHONY: directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Build static library
$(LIBRARY): $(LIBRARY_OBJ)
	@echo "Creating static library: $@"
	ar rcs $@ $^

# Compile library object file
$(LIBRARY_OBJ): $(LIBRARY_SRC) enhanced_marketing_agent.hpp
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile demo object file
$(DEMO_OBJ): $(DEMO_SRC) enhanced_marketing_agent.hpp
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link demo executable
$(DEMO_BIN): $(DEMO_OBJ) $(LIBRARY)
	@echo "Linking: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@

# Run the demo
.PHONY: run
run: $(DEMO_BIN)
	@echo ""
	@echo "Running Enhanced Marketing Agent Demo..."
	@echo ""
	./$(DEMO_BIN)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Clean complete!"

# Clean and rebuild
.PHONY: rebuild
rebuild: clean all

# Install (copy to system directories - requires sudo)
.PHONY: install
install: release
	@echo "Installing Enhanced Marketing Agent..."
	install -d /usr/local/lib
	install -d /usr/local/include
	install -d /usr/local/bin
	install -m 644 $(LIBRARY) /usr/local/lib/
	install -m 644 enhanced_marketing_agent.hpp /usr/local/include/
	install -m 755 $(DEMO_BIN) /usr/local/bin/
	@echo "Installation complete!"

# Uninstall
.PHONY: uninstall
uninstall:
	@echo "Uninstalling Enhanced Marketing Agent..."
	rm -f /usr/local/lib/libmarketing_agent.a
	rm -f /usr/local/include/enhanced_marketing_agent.hpp
	rm -f /usr/local/bin/marketing_demo
	@echo "Uninstall complete!"

# Display help
.PHONY: help
help:
	@echo ""
	@echo "═══════════════════════════════════════════════════════"
	@echo "  Enhanced Marketing Agent - Makefile Help"
	@echo "═══════════════════════════════════════════════════════"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build in release mode (default)"
	@echo "  make release      - Build optimized release version"
	@echo "  make debug        - Build with debug symbols"
	@echo "  make run          - Build and run the demo"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make rebuild      - Clean and rebuild"
	@echo "  make install      - Install to system (requires sudo)"
	@echo "  make uninstall    - Remove from system (requires sudo)"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make && make run  - Quick build and run"
	@echo "  make debug run    - Debug build and run"
	@echo ""
	@echo "═══════════════════════════════════════════════════════"
	@echo ""

# Show build info
.PHONY: info
info:
	@echo ""
	@echo "═══════════════════════════════════════════════════════"
	@echo "  Build Information"
	@echo "═══════════════════════════════════════════════════════"
	@echo "  Compiler:       $(CXX)"
	@echo "  C++ Standard:   C++17"
	@echo "  Flags:          $(CXXFLAGS)"
	@echo "  Build dir:      $(BUILD_DIR)"
	@echo "  Binary dir:     $(BIN_DIR)"
	@echo "═══════════════════════════════════════════════════════"
	@echo ""
