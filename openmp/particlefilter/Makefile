include ../Make.inc

EXE  = particle_filter

.PHONY: all
all: $(EXE)

$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE)
