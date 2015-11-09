include ../common.mk

EXE  = backprop_cuda
OBJS = backprop.o facetrain.o imagenet.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.dat
