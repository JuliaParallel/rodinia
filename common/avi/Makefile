include ../../common.mk

OBJS = avilib.o avimod.o
LIB = libavi.so

CFLAGS += -fPIC

.PHONY: all
all: $(LIB)

$(LIB): $(OBJS)
	$(CC) -shared -o $(LIB) $(OBJS)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(LIB)
