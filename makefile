TARGET ?= cpuinfo
ARCH ?= native
OUTDIR ?= out

ifeq ($(ARCH),native)
ARCH_FLAGS =
else ifeq ($(ARCH),x86)
ARCH_FLAGS = -m32
else ifeq ($(ARCH),x64)
ARCH_FLAGS = -m64
else ifeq ($(ARCH),armv7)
CROSS_COMPILE ?= arm-linux-gnueabihf-
ARCH_FLAGS = -march=armv7-a -mfpu=neon -mfloat-abi=hard
else ifeq ($(ARCH),arm64)
CROSS_COMPILE ?= aarch64-linux-gnu-
ARCH_FLAGS = -march=armv8-a
else
$(error Unsupported ARCH '$(ARCH)': native x86 x64 armv7 arm64)
endif

CROSS_COMPILE ?=
ifeq ($(origin CC),default)
CC = $(CROSS_COMPILE)gcc
endif
ifeq ($(origin CXX),default)
CXX = $(CROSS_COMPILE)g++
endif

CPPFLAGS += -D_GNU_SOURCE -Isrc
CFLAGS ?= -O3 -Wall -std=c99
CXXFLAGS ?= -O3 -Wall -std=c++11
LDFLAGS ?= -static
LDLIBS ?=

BUILD_DIR = $(OUTDIR)/$(ARCH)
BINARY = $(BUILD_DIR)/$(TARGET)
C_FILES = $(wildcard *.c src/*.c)
CXX_FILES = $(wildcard *.cpp src/*.cpp)
OBJ_FILES = $(addprefix $(BUILD_DIR)/, $(C_FILES:.c=.o) $(CXX_FILES:.cpp=.o))
DEP_FILES = $(OBJ_FILES:.o=.d)

.PHONY: all clean
all: $(BINARY)

$(BINARY): $(OBJ_FILES) makefile
	$(CXX) $(ARCH_FLAGS) $(LDFLAGS) -o $@ $(OBJ_FILES) $(LDLIBS)

$(BUILD_DIR)/%.o: %.c makefile
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(ARCH_FLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp makefile
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(ARCH_FLAGS) -MMD -MP -c $< -o $@

clean:
	rm -rf -- $(BUILD_DIR)

-include $(DEP_FILES)
