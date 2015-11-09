include ../common.mk

EXE  = b+tree
OBJS = kernel/kernel_cpu.o kernel/kernel_cpu_2.o util/timer/timer.o util/num/num.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
