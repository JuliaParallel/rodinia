include ../Make.inc

EXE  = leukocyte
OBJS = avilib.o find_ellipse.o track_ellipse.o misc_math.o $(MESCHACH_DIR)/meschach.a

.PHONY: all
all: $(EXE)

MESCHACH_DIR=../../common/meschach
$(MESCHACH_DIR)/Makefile:
	cd $(MESCHACH_DIR); ./configure --with-all
$(MESCHACH_DIR)/meschach.a: $(MESCHACH_DIR)/Makefile
	$(MAKE) -C $(MESCHACH_DIR) all
CPPFLAGS += -I$(MESCHACH_DIR)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
	$(MAKE) -C $(MESCHACH_DIR) clean
