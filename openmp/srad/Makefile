.PHONY: all
all: srad_v1 srad_v2

.PHONY: srad_v1 srad_v2
srad_v1 srad_v2:
	$(MAKE) -C $@

.PHONY: clean
clean:
	$(MAKE) -C srad_v1 $@
	$(MAKE) -C srad_v2 $@