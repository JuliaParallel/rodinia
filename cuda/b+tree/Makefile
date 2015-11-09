include ../common.mk

EXE  = b+tree
OBJS = kernel/kernel_gpu_cuda_wrapper.o kernel/kernel_gpu_cuda_wrapper_2.o util/timer/timer.o util/num/num.o util/cuda/cuda.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm -lstdc++ 

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
