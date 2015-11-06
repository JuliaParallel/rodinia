OPENMP_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
include $(OPENMP_DIR)/../Make.inc

CFLAGS   += -fopenmp
CXXFLAGS += -fopenmp
LDLIBS   += -fopenmp
