include ../common.mk

EXE  = b+tree
OBJS = kernel/kernel_gpu_opencl_wrapper.o kernel/kernel_gpu_opencl_wrapper_2.o util/timer/timer.o util/num/num.o util/opencl/opencl.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
