include ../common.mk

EXE  = lud
OBJS = common/common.o lud_kernel.o

CPPFLAGS += -Icommon

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
