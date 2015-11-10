include ../common.mk

EXE  = lavaMD
OBJS = kernel/kernel_gpu_cuda_wrapper.o util/num/num.o util/timer/timer.o util/device/device.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
