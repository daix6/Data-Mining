CC = g++

SOURCE_DIR = src
OBJECT_DIR = obj
BUILD_DIR = build

_OBJS = main.o
OBJS = $(patsubst %, $(OBJECT_DIR)/%, $(_OBJS))

CFLAGS = -Wall -g -std=c++11 -fopenmp -lm
LFLAGS = -lgomp

ifeq ($(OS), Windows_NT)
		EXECUTABLE = main.exe
else
		EXECUTABLE = main
endif

all: $(EXECUTABLE)

$(OBJECT_DIR)/%.o: $(SOURCE_DIR)/%.cpp
		$(CC) $(CFLAGS) -c $< -o $@

$(EXECUTABLE): $(OBJS)
		$(CC) -o $@ $^ $(LFLAGS)

.PHONY: clean

clean:
		rm -rf $(OBJECT_DIR)/*.o $(EXECUTABLE) dest/theta*.csv build/*.csv