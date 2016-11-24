%: %.xz
	xz -d -k $<

DATASETS = output.txt

.PHONY: all
all: $(DATASETS)

.PHONY: clean
clean:
	$(RM) $(DATASETS)
