OPENMP_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
include $(OPENMP_DIR)/../common.mk

LDLIBS   += -lOpenCL
