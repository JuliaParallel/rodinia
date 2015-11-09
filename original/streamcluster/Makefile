include ../../common.mk

EXE  = streamcluster

CPPFLAGS += -DENABLE_THREADS
LDFLAGS  += -pthread

.PHONY: all
all: $(EXE)

.PHONY: clean
clean:
	$(RM) $(EXE)
