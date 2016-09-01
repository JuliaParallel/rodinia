include ../common.mk

EXE  = leukocyte
OBJS = find_ellipse.o find_ellipse_opencl.o track_ellipse.o track_ellipse_opencl.o misc_math.o OpenCL_helper_library.o \
       $(MESCHACH_DIR)/meschach.a \
       $(AVI_DIR)/avilib.o

.PHONY: all
all: $(EXE)

MESCHACH_DIR=../../common/meschach
$(MESCHACH_DIR)/Makefile:
	cd $(MESCHACH_DIR); ./configure --with-all
$(MESCHACH_DIR)/meschach.a: $(MESCHACH_DIR)/Makefile
	$(MAKE) -C $(MESCHACH_DIR) all
CPPFLAGS += -I$(MESCHACH_DIR)

AVI_DIR=../../common/avi
$(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o:
	$(MAKE) -C $(AVI_DIR)
CPPFLAGS += -I$(AVI_DIR)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
	-$(MAKE) -C $(MESCHACH_DIR) realclean
	$(MAKE) -C $(AVI_DIR) clean
