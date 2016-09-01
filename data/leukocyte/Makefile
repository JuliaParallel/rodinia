include ../common.mk

DATASET = testfile.avi

$(DATASET): $(sort $(wildcard $(DATASET).part*))
	cat $^ > $@

.PHONY: clean
clean:
	$(RM) $(DATASET)
