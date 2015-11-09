include ../../common.mk

EXE  = lud
OBJS = lud_base.o ../common/common.o

CPPFLAGS += -I../common

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
