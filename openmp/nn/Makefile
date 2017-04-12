include ../common.mk

EXES = nn

.PHONY: all
all: $(EXES)

$(EXES): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXES) output.txt
