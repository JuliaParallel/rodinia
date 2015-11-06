include ../Make.inc

EXE  = lavaMD
OBJS = kernel/kernel_gpu_opencl_wrapper.o util/num/num.o util/timer/timer.o util/opencl/opencl.o

ifdef OUTPUT
CPPFLAGS += -DOUTPUT
endif

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
