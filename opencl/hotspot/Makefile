include ../common.mk

EXE  = hotspot
OBJS = OpenCL_helper_library.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
