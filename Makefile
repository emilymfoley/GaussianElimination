# Compiler flags
CFLAGS = -std=c99            # Use the C99 standard
CFLAGS += -fsignaling-nans    # Enable signaling NaNs
CFLAGS += -g -ggdb3           # Enable debugging information
CFLAGS += -O5                 # Optimization level 5

# Linker flags
LDFLAGS = -lm                 # Link against the math library

# Python executable
PYTHON = python               # Name of the Python executable

# List of object files
OBJS = gauss_solve.o main.o helpers.o

# Build targets
all: gauss_solve libgauss.so

# Dependency rules
gauss_solve.o: gauss_solve.h
helpers.o: helpers.h

# Link object files to create the executable
gauss_solve: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

# Check targets
check: check_gauss_solve check_ctype_wrapper

check_gauss_solve: gauss_solve
	./$<

check_ctype_wrapper: gauss_solve.py libgauss.so
	$(PYTHON) ./$<

# Build the shared library
LIB_SOURCES = gauss_solve.c
libgauss.so: $(LIB_SOURCES)
	gcc -shared -I/usr/include/python3.12 -o $@ -fPIC $(LIB_SOURCES)

# Clean up files
clean: FORCE
	@-rm gauss_solve *.o
	@-rm *.so

# Force target to handle phony rules
FORCE:
