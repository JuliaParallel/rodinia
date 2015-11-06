include ../Make.inc

EXE  = kmeans
OBJS = cluster.o getopt.o kmeans.o kmeans_clustering.o 

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
