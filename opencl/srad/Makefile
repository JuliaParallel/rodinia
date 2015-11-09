include ../common.mk

EXE  = srad
OBJS = kernel/kernel_gpu_opencl_wrapper.o util/opencl/opencl.o util/graphics/graphics.o util/graphics/resize.o util/timer/timer.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
