include ../common.mk

EXE  = streamcluster
OBJS = streamcluster_cuda.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
