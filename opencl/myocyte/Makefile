include ../common.mk

EXE  = myocyte
OBJS = kernel/kernel_gpu_opencl_wrapper.o util/file/file.o util/num/num.o util/timer/timer.o util/opencl/opencl.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
