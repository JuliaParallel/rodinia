include common.mk

BENCHMARKS := $(sort $(dir $(wildcard */)))

.PHONY: all
all: $(BENCHMARKS)

.PHONY: $(BENCHMARKS)
$(BENCHMARKS):
	$(MAKE) -C $@

.PHONY: clean
clean:
	for BENCHMARK in $(BENCHMARKS); do $(MAKE) -C $$BENCHMARK clean; done
