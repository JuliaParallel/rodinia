include ../common.mk

.PHONY: all
all: inputGen/hurricanegen
	inputGen/hurricanegen 655360 64

inputGen/hurricanegen:
	$(MAKE) -C inputGen

.PHONY: clean
clean:
	$(RM) cane* list*
	$(MAKE) -C inputGen clean
