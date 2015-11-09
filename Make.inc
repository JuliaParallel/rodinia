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
