include ../common.mk

DATASET = test.avi

$(DATASET): $(sort $(wildcard $(DATASET).part*))
	cat $^ > $@

.PHONY: clean
clean:
	$(RM) $(DATASET)
