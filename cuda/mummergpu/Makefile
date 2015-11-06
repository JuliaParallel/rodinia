include ../Make.inc

EXE  = mummergpu
OBJS = mummergpu_gold.o mummergpu_cuda.o suffix-tree.o PoolMalloc.o

ifdef OUTPUT
CPPFLAGS += -DOUTPUT
endif

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm -fopenmp -lcuda

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
