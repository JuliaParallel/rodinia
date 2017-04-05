SHELL=/bin/sh -ue

CFLAGS   += -O2
CXXFLAGS += -O2

ifdef OUTPUT
CPPFLAGS += -DOUTPUT
endif

ifdef DEBUG
CFLAGS   += -g
CXXFLAGS += -g
endif

# include Make.user relative to every active Makefile
$(foreach MAKEFILE,$(MAKEFILE_LIST),\
	$(eval -include $(shell dirname $(realpath $(MAKEFILE)))/Make.user)\
)
