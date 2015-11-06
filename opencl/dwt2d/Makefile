include ../Make.inc

EXE  = dwt2d
OBJS = components.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

ifdef OUTPUT
CPPFLAGS += -DOUTPUT
endif

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
