include ../Make.inc

EXE  = lavaMD
OBJS = kernel/kernel_gpu_cuda_wrapper.o util/num/num.o util/timer/timer.o util/device/device.o

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
