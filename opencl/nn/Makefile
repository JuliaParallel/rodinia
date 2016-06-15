include ../common.mk

EXE  = nn
OBJS = clutils.o utils.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
