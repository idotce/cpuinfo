PWDDIR = $(shell pwd)
TARGET = cpuinfo
OUTDIR = out
ARCH ?=
CROSS_COMPILE ?=

MKDIR = mkdir
RMDIR = rm -rf  #RMDIR = rmdir /q /s

CC = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++
LD = $(CROSS_COMPILE)g++

DEFINES = \
	-D_GNU_SOURCE

INCLUDES = \
	-Isrc

LIBS =
#LIBS = -lgdi32 -lopengl32 -lglu32

CFLAGS = $(DEFINES) $(INCLUDES) \
	-O3 -Wall -std=c99

CXXFLAGS = $(DEFINES) $(INCLUDES) \
	-O3 -Wall -std=c++11

LDFLAGS = $(LIBS) -static

ifneq ($(ARCH),)
	include makefile.$(ARCH)
endif

C_FILES = $(wildcard *.c src/*.c)
CXX_FILES = $(wildcard *.cpp src/*.cpp)
OBJ_FILES = $(addprefix $(OUTDIR)/, $(C_FILES:.c=.o) $(CXX_FILES:.cpp=.o))

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(LD) -o $(OUTDIR)/$@ $^ $(LDFLAGS)

$(OUTDIR)/%.o: %.c
	@${MKDIR} -p ${dir ${@}}
	$(CC) -o $@ -c $< $(CFLAGS)

$(OUTDIR)/%.o: %.cpp
	@${MKDIR} -p ${dir ${@}}
	$(CXX) -o $@ -c $< $(CXXFLAGS)

clean:
	@${RMDIR} ${OUTDIR}

install:
	@ERROR
