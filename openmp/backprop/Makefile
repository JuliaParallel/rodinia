include ../common.mk

EXE  = backprop
OBJS = facetrain.o imagenet.o backprop_kernel.o

EXE2  = backprop1d
OBJS2 = facetrain.o imagenet.o backprop_kernel1d.o

.PHONY: all
all: $(EXE) $(EXE2)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

$(EXE2): $(OBJS2)
$(EXE2): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.dat
	$(RM) $(EXE2) $(OBJS2) output.dat
