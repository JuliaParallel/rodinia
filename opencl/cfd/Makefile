include ../../common/make.config
#C compiler
CC = g++

#OPENCL_DIR = /home/sc5nf/ati-stream-sdk-v2.2-lnx64/
#INC = $(OPENCL_DIR)/include/ 
#LIB = $(OPENCL_DIR)/lib/x86_64/

FLAGS = -O3
SRC = euler3d.cpp

EXE = euler3d

$(EXE): $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL $(FLAGS) 

.PHONY: clean
clean: $(SRC)
	$(RM) $(EXE) $(EXE).linkinfo result*
