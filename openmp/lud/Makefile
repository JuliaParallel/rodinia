include ../common.mk

EXE  = lud
OBJS = lud_omp.o common/common.o

CPPFLAGS += -Icommon

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
