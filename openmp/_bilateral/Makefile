include ../common.mk

EXE  = bilateralFilter
OBJS = bilateralFilter_cpu.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.dat
