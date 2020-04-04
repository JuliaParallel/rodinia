include ../common.mk

EXE  = myocyte myocyte1d

.PHONY: all
all: $(EXE)

$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE)
