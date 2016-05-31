include common.mk

SUITES := openmp cuda opencl julia_cuda julia_st

.PHONY: compile
compile: $(SUITES)

.PHONY: $(SUITES)
$(SUITES):
	$(MAKE) -C $@

.PHONY: clean
clean:
	for SUITE in $(SUITES); do $(MAKE) -C $$SUITE clean; done
