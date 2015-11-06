include ../Make.inc

EXE  = backprop
OBJS = backprop.o facetrain.o imagenet.o backprop_kernel.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
