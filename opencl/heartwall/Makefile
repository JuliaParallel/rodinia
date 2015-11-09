include ../common.mk

EXE  = heartwall
OBJS = kernel/kernel_gpu_opencl_wrapper.o util/avi/avilib.o util/avi/avimod.o util/file/file.o util/timer/timer.o util/opencl/opencl.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
