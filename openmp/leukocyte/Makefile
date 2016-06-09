include ../common.mk

EXE  = leukocyte
OBJS = $(MESCHACH_DIR)/libmeschach.a find_ellipse.o track_ellipse.o misc_math.o $(MESCHACH_DIR)/libmeschach.a  \
       $(AVI_DIR)/avilib.o

.PHONY: all
all: $(EXE)

MESCHACH_DIR=../../common/meschach
$(MESCHACH_DIR)/Makefile:
	cd $(MESCHACH_DIR); ./configure --with-all
$(MESCHACH_DIR)/libmeschach.a: $(MESCHACH_DIR)/Makefile
	$(MAKE) -C $(MESCHACH_DIR) all
CPPFLAGS += -I$(MESCHACH_DIR)

AVI_DIR=../../common/avi
$(AVI_DIR)/avilib.o $(AVI_DIR)/avimod.o:
	$(MAKE) -C $(AVI_DIR)
CPPFLAGS += -I$(AVI_DIR)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm $(MESCHACH_DIR)/libmeschach.a

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
	-$(MAKE) -C $(MESCHACH_DIR) clean
	$(MAKE) -C $(AVI_DIR) clean
