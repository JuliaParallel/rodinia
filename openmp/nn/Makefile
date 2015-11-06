include ../Make.inc

EXES = nn hurricane_gen

.PHONY: all
all: $(EXES)

$(EXES): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXES)
