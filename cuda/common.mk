CUDA_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
include $(CUDA_DIR)/../common.mk


#
# Auxiliary
#

DUMMY=
SPACE=$(DUMMY) $(DUMMY)
COMMA=$(DUMMY),$(DUMMY)

define join-list
$(subst $(SPACE),$(2),$(1))
endef


#
# CUDA detection
#

CUDA_ROOT = /usr

MACHINE := $(shell uname -m)
ifeq ($(MACHINE), x86_64)
LDFLAGS += -L$(CUDA_ROOT)/lib64
endif
ifeq ($(MACHINE), i686)
LDFLAGS += -L$(CUDA_ROOT)/lib
endif

CPPFLAGS += -I$(CUDA_ROOT)/include

NVCC=$(CUDA_ROOT)/bin/nvcc

LDLIBS   += -lcudart


#
# NVCC compilation
#

# NOTE: passing -lcuda to nvcc is redundant
NONCUDA_LDLIBS = $(filter-out -lcuda -lcudart,$(LDLIBS))

ifneq ($(strip $(NONCUDA_LDLIBS)),)
NVCC_LDLIBS = -Xcompiler $(call join-list,$(NONCUDA_LDLIBS),$(COMMA))
endif

NVCCFLAGS += -arch=sm_35

%: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $^

%.o: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c -o $@ $<
