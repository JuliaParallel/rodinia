include ../common.mk

EXE  = kmeans
OBJS = cluster.o getopt.o kmeans.o kmeans_clustering.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
