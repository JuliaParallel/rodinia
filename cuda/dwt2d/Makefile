include ../common.mk

EXE  = dwt2d
OBJS = dwt.o components.o dwt_cuda/common.o dwt_cuda/fdwt53.o dwt_cuda/fdwt97.o dwt_cuda/rdwt53.o dwt_cuda/rdwt97.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
