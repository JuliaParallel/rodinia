include ../Make.inc

EXE  = kmeans
OBJS = cluster.o getopt.o read_input.o kmeans_clustering.o  rmse.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
$(EXE): LDLIBS += -lm

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS)
