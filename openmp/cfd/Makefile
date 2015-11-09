include ../common.mk

EXES = euler3d_cpu euler3d_cpu_double pre_euler3d_cpu pre_euler3d_cpu_double

OMP_NUM_THREADS = 8

.PHONY: all
all: $(EXES)

.PHONY: clean
clean:
	$(RM) $(EXES) density momentum density_energy
