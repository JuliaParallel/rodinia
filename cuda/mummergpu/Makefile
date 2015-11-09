include ../common.mk

EXE  = mummergpu
OBJS = mummergpu_gold.o mummergpu_cuda.o suffix-tree.o PoolMalloc.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm -fopenmp -lcuda

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
