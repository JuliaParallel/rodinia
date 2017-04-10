OBJS = $(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o

.PHONY: all
all: $(OBJS)

# we call make recursively, so need to export variables
export

AVI_DIR=../../common/avi
$(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o:
	$(MAKE) -C $(AVI_DIR)
CPPFLAGS += -I$(AVI_DIR)

.PHONY: clean
clean:
	$(RM) $(OBJS) output.txt
	$(MAKE) -C $(AVI_DIR) clean
