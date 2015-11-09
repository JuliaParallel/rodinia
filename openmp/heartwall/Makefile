include ../common.mk

EXE  = heartwall
OBJS = AVI/avilib.o AVI/avimod.o

CPPFLAGS += -I./AVI

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
