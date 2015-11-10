include ../common.mk

EXE  = lavaMD
OBJS = kernel/kernel_cpu.o util/num/num.o util/timer/timer.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
