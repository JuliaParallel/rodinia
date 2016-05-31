include common.mk

SUITES := openmp cuda opencl julia_cuda

.PHONY: compile
compile: $(SUITES)

.PHONY: $(SUITES)
$(SUITES):
	$(MAKE) -C $@

.PHONY: clean
clean:
	for SUITE in $(SUITES); do $(MAKE) -C $$SUITE clean; done
