include ../Make.inc

EXE  = lud
OBJS = common/common.o

CPPFLAGS += -Icommon

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
