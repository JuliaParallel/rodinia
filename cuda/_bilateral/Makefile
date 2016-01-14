include ../common.mk

EXE  = bilateralFilter
OBJS = bmploader.o bilateral_kernel.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.ppm
