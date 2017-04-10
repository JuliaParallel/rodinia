include ../common.mk

EXE  = leukocyte
OBJS = find_ellipse.o find_ellipse_kernel.o track_ellipse.o track_ellipse_kernel.o misc_math.o \
       $(MESCHACH_DIR)/libmeschach.a \
       $(AVI_DIR)/avilib.o

.PHONY: all
all: $(EXE)

# we call make recursively, so need to export variables
export

MESCHACH_DIR = ../../common/meschach
$(MESCHACH_DIR)/makefile:
	cd $(MESCHACH_DIR); ./configure --with-all
$(MESCHACH_DIR)/libmeschach.a: $(MESCHACH_DIR)/makefile
	$(MAKE) -C $(MESCHACH_DIR)
CPPFLAGS += -I$(MESCHACH_DIR)
misc_math.o: $(MESCHACH_DIR)/libmeschach.a
find_ellipse.o: $(MESCHACH_DIR)/libmeschach.a
track_ellipse.o: $(MESCHACH_DIR)/libmeschach.a
track_ellipse_kernel.o: $(MESCHACH_DIR)/libmeschach.a

AVI_DIR=../../common/avi
$(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o:
	$(MAKE) -C $(AVI_DIR)
CPPFLAGS += -I$(AVI_DIR)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm -lstdc++

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
	-$(MAKE) -C $(MESCHACH_DIR) realclean
	$(MAKE) -C $(AVI_DIR) clean
