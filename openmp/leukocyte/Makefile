OUTPUT := Y

include ../common.mk

MESCHACH_DIR=../../common/meschach
AVI_DIR=../../common/avi

EXE  = leukocyte
OBJS = find_ellipse.o track_ellipse.o misc_math.o \
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

$(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o:
	$(MAKE) -C $(AVI_DIR)
CPPFLAGS += -I$(AVI_DIR)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm $(MESCHACH_DIR)/libmeschach.a

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) result.txt
	-$(MAKE) -C $(MESCHACH_DIR) realclean
	$(MAKE) -C $(AVI_DIR) clean
