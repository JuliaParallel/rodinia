include common.mk

SUITES := data openmp cuda opencl julia_cuda julia_st results

.PHONY: compile
compile: $(SUITES)

.PHONY: $(SUITES)
$(SUITES):
	$(MAKE) -C $@

.PHONY: clean
clean:
	for SUITE in $(SUITES); do $(MAKE) -C $$SUITE clean; done
