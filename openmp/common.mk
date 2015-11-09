OPENMP_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
include $(OPENMP_DIR)/../common.mk

CFLAGS   += -fopenmp
CXXFLAGS += -fopenmp
LDLIBS   += -fopenmp
