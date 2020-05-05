OPENMP_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
include $(OPENMP_DIR)/../common.mk

OMPFLAGS = -fopenmp -I/home/pschen/llvm/thesis/build-Debug/include

CFLAGS   += $(OMPFLAGS)
CXXFLAGS   += $(OMPFLAGS) -D__FUCK_FOR_THESIS__
LDLIBS   += $(OMPFLAGS)

ifdef OFFLOAD
OMPOFFLOADFLAGS = -DOMP_OFFLOAD -fopenmp-targets=nvptx64
CXXFLAGS += $(OMPOFFLOADFLAGS)
CFLAGS += $(OMPOFFLOADFLAGS)
LDLIBS += $(OMPOFFLOADFLAGS)
endif

ifdef DC
OMPDCFLAGS = -DOMP_DC
CXXFLAGS += $(OMPDCFLAGS)
CFLAGS += $(OMPDCFLAGS)
endif

ifdef PG
PGFLAGS = -pg
CXXFLAGS += $(PGFLAGS)
CFLAGS += $(PGFLAGS)
endif
