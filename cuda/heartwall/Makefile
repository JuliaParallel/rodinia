include ../Make.inc

EXE  = heartwall
OBJS = AVI/avilib.o AVI/avimod.o

CPPFLAGS += -I./AVI

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
