CXX=g++
CFLAGS = -Wall -Ilibs/onnxruntime/include 
LDFLAGS = -Llibs/onnxruntime/lib
LDLIBS = -lonnxruntime -lstdc++

SRCS = main.cpp 
OBJS = $(SRCS:.cpp=.o)
EXEC = main 

all: $(EXEC)

$(EXEC): $(OBJS)
    $(CXX) $(LDFLAGS) $(OBJS) -o $(EXEC) $(LDLIBS)

%.o: %.cpp
    $(CXX) $(CFLAGS) -c $< -o $@

clean:
    $(RM) $(OBJS) $(EXEC)
