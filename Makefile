SHELL=sh -ue

include common/$(MAKE).config

RODINIA_BASE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

OPENMP_DIRS  := backprop bfs cfd          heartwall hotspot kmeans lavaMD leukocyte lud nn nw srad streamcluster particlefilter pathfinder
CUDA_DIRS    := backprop bfs cfd gaussian heartwall hotspot kmeans lavaMD leukocyte lud nn nw srad streamcluster particlefilter pathfinder mummergpu hybridsort dwt2d
OPENCL_DIRS  := backprop bfs cfd gaussian heartwall hotspot kmeans lavaMD leukocyte lud nn nw srad streamcluster particlefilter pathfinder           hybridsort dwt2d


.PHONY: compile
compile: compile_cuda compile_openmp compile_opencl

.PHONY: compile_openmp
compile_openmp:	
	for dir in $(OPENMP_DIRS); do $(MAKE) -C "openmp/$$dir"; done

.PHONY: compile_cuda
compile_cuda:
	for dir in $(CUDA_DIRS);   do $(MAKE) -C "cuda/$$dir";   done

.PHONY: compile_opencl
compile_opencl:
	for dir in $(OPENCL_DIRS); do $(MAKE) -C "opencl/$$dir"; done


.PHONY: clean
clean: clean_cuda clean_openmp clean_opencl

.PHONY: clean_cuda
clean_cuda:
	for dir in $(CUDA_DIRS);   do $(MAKE) -C "cuda/$$dir" clean;   done

.PHONY: clean_OMP
clean_openmp:
	for dir in $(OPENMP_DIRS); do $(MAKE) -C "openmp/$$dir" clean; done

.PHONY: clean_opencl
clean_opencl:
	for dir in $(OPENCL_DIRS); do $(MAKE) -C "opencl/$$dir" clean; done
