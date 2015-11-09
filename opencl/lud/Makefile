include ../common.mk

EXE  = lud
OBJS = common/common.o

CPPFLAGS += -Icommon

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
