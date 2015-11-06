include ../Make.inc

EXE  = backprop_ocl
OBJS = backprop.o facetrain.o imagenet.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
