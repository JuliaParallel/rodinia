include ../common.mk

EXE  = heartwall
OBJS = kernel/kernel_gpu_opencl_wrapper.o util/file/file.o util/timer/timer.o util/opencl/opencl.o \
       $(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o

.PHONY: all
all: $(EXE)

AVI_DIR=../../common/avi
$(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o:
	$(MAKE) -C $(AVI_DIR)
CPPFLAGS += -I$(AVI_DIR)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
	$(MAKE) -C $(AVI_DIR) clean
