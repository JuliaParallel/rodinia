ROOT := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
include $(ROOT)/../Make.inc

CFLAGS   += -fopenmp
CXXFLAGS += -fopenmp
LDLIBS   += -fopenmp
