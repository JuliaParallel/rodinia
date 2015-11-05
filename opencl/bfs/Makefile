include ../../common/make.config

#C compiler
CC = g++

SRC = bfs.cpp timer.cc

CC_FLAGS = -g -O3

EXE = bfs

$(EXE): $(SRC)
	$(CC) $(CC_FLAGS)  -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 

.PHONY: clean
clean: $(SRC)
	$(RM) $(EXE) $(EXE).linkinfo result*
