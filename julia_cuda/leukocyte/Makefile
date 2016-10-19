OUTPUT := Y

include ../../openmp/common.mk

INPUT_FILE_DIR=../../data/leukocyte/
INPUT_FILE=$(INPUT_FILE_DIR)/testfile.avi

AVI_DIR=../../common/avi

EXE  =
OBJS = $(AVI_DIR)/libavi.so $(INPUT_FILE)

.PHONY: all

all: $(OBJS)

$(INPUT_FILE):
	$(MAKE) -C $(INPUT_FILE_DIR)

$(AVI_DIR)/libavi.so:
	$(MAKE) -C $(AVI_DIR)

.PHONY: clean
clean:
	$(RM) result.txt
	$(MAKE) -C $(AVI_DIR) clean
