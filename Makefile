include Make.inc

SUITES := openmp cuda opencl

.PHONY: compile
compile: $(SUITES)

.PHONY: $(SUITES)
$(SUITES):
	$(MAKE) -C $@

.PHONY: clean
clean:
	for SUITE in $(SUITES); do $(MAKE) -C $$SUITE clean; done
