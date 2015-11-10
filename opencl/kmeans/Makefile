include ../common.mk

EXE  = kmeans
OBJS = cluster.o getopt.o read_input.o kmeans_clustering.o  rmse.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
