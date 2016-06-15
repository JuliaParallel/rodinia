EXES = hurricane_gen

.PHONY: all
all: $(EXES)

$(EXES): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXES) output.txt
