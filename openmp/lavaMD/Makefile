include ../Make.inc

EXE  = lavaMD
OBJS = lavaMD.o kernel/kernel_cpu.o util/num/num.o util/timer/timer.o

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
