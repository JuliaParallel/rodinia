include ../common.mk

EXE  = kmeans
OBJS = cluster.o getopt.o kmeans.o kmeans_clustering.o

EXE2 = kmeans1d
OBJS2 = cluster1d.o getopt.o kmeans1d.o kmeans_clustering1d.o

.PHONY: all
all: $(EXE) $(EXE2)

$(EXE): $(OBJS)
$(EXE2): $(OBJS2)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
	$(RM) $(EXE2) $(OBJS2) output.txt
