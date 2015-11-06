include ../Make.inc

EXE  = heartwall
OBJS = kernel/kernel_gpu_opencl_wrapper.o util/avi/avilib.o util/avi/avimod.o util/file/file.o util/timer/timer.o util/opencl/opencl.o

ifdef OUTPUT
CPPFLAGS += -DOUTPUT
endif

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
