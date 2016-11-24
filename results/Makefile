BENCHMARKS := srad_v2

.PHONY: all
all: $(BENCHMARKS)

.PHONY: $(BENCHMARKS)
$(BENCHMARKS):
	$(MAKE) -C $@ -f ../common.mk

.PHONY: clean
clean:
	for BENCHMARK in $(BENCHMARKS); \
		do $(MAKE) -C $$BENCHMARK -f ../common.mk clean; done
