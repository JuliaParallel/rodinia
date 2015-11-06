include ../Make.inc

EXE  = myocyte

.PHONY: all
all: $(EXE)

$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE)
