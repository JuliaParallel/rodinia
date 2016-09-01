BENCHMARKS := bfs cfd heartwall hotspot kmeans leukocyte lud mummergpu

.PHONY: all
all: $(BENCHMARKS)

.PHONY: $(BENCHMARKS)
$(BENCHMARKS):
	$(MAKE) -C $@

.PHONY: clean
clean:
	for BENCHMARK in $(BENCHMARKS); do $(MAKE) -C $$BENCHMARK clean; done
