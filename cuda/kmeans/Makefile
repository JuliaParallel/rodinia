include ../common.mk

EXE  = kmeans_cuda
OBJS = cluster.o kmeans.o kmeans_clustering.o rmse.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
