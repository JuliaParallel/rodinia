OPENMP_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
include $(OPENMP_DIR)/../common.mk

OMPFLAGS = -fopenmp

CFLAGS   += $(OMPFLAGS)
CXXFLAGS += $(OMPFLAGS)
LDLIBS   += $(OMPFLAGS)

ifdef OFFLOAD
CXXFLAGS += -DOMP_OFFLOAD
endif
