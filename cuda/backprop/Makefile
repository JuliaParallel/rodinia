include ../Make.inc

EXE  = backprop_cuda
OBJS = backprop.o facetrain.o imagenet.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
