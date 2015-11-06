include ../../common/make.config

#C compiler
CC = g++


FLAGS = -O3
SRC = streamcluster.cpp
EXE = streamcluster

$(EXE): $(SRC)
	$(CC) -lOpenCL  $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB)	$(FLAGS)
	
clean: $(SRC)
	$(RM) $(EXE) $(EXE).linkinfo result*
