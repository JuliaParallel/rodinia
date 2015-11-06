include ../Make.inc

EXE  = hybridsort
OBJS = bucketsort.o mergesort.o

.PHONY: all
all: $(EXE)

ifdef VERIFY
CPPFLAGS += VERIFY = -DVERIFY
endif

ifdef OUTPUT
CPPFLAGS += -DOUTPUT
endif

ifdef TIMER
CPPFLAGS += TIMER = -DTIMER
endif

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
