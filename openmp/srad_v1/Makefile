include ../common.mk

EXE  = srad

.PHONY: all
all: $(EXE)

$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE)
