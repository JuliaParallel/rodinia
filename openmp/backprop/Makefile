include ../common.mk

EXE  = backprop
OBJS = facetrain.o imagenet.o backprop_kernel.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.dat
