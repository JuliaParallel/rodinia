include ../Make.inc

EXE  = heartwall
OBJS = AVI/avilib.o AVI/avimod.o

CPPFLAGS += -I./AVI
ifdef OUTPUT
CPPFLAGS += -DOUTPUT
endif

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
