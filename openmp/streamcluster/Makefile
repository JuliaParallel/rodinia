PREFIX=${PARSECDIR}/pkgs/kernels/streamcluster/inst/${PARSECPLAT}

TARGET_C = sc_cpu

TARGET_O = sc_omp

ifdef version
  ifeq "$(version)" "parallel"
    CXXFLAGS :=	$(CXXFLAGS) -DENABLE_THREADS -pthread
  endif
endif

all: sc_cpu sc_omp

sc_cpu:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) streamcluster_original.cpp -o $(TARGET_C) $(LIBS) -DENABLE_THREADS -pthread

sc_omp:
	g++ -O3 -fopenmp -o $(TARGET_O) streamcluster_omp.cpp


clean:
	$(RM) -r *.o *~ *.txt sc_cpu sc_omp
	
