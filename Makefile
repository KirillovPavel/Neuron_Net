CC		:= g++
TARGET		:= main
FLAGS		:= -std=c++17
LIBS		:=
SOURCE		:= $(wildcard src/*.cpp import/*.cpp context/*.cpp)
HEADERS		:= $(wildcard src/*.hpp import/*.hpp context/*.hpp)
OBJS		:= $(patsubst %.cpp, %.o, $(SOURCE))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LIBS)

%.o: %.cpp $(HEADERS)
	$(CC) -o $@ -c $< $(FLAGS)

.PHONY: all clean debug
