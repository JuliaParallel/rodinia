include ../Make.inc

EXES = euler3d_cpu euler3d_cpu_double pre_euler3d_cpu pre_euler3d_cpu_double

OMP_NUM_THREADS = 8
CPPFLAGS += -Dblock_length=$(OMP_NUM_THREADS)

.PHONY: all
all: $(EXES)

.PHONY: clean
clean:
	$(RM) $(EXES)
