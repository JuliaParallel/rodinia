OUTPUT := Y

include ../../openmp/common.mk

AVI_DIR=../../common/avi

EXE  =
OBJS = $(AVI_DIR)/libavi.so

.PHONY: all
all: $(OBJS)

$(AVI_DIR)/libavi.so:
	$(MAKE) -C $(AVI_DIR)

.PHONY: clean
clean:
	$(RM) result.txt
	$(MAKE) -C $(AVI_DIR) clean
