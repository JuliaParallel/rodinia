include ../Make.inc

EXE  = lud
OBJS = common/common.o lud_kernel.o

CPPFLAGS += -Icommon

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
