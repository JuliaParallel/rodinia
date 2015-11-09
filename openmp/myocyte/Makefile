include ../common.mk

EXE  = myocyte

.PHONY: all
all: $(EXE)

$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE)
