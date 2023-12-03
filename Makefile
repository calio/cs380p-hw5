# Makefile for MPI program

CC=mpicc               # Compiler to use
CFLAGS=-Wall -O2       # Compiler flags, e.g., optimization and warning flags
TARGET=mpi_hello_world # Target executable name
SRC=main.cpp  # Source file

all: $(TARGET)

#$(TARGET): $(SRC)
#	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

bh: bh.cpp
	g++ $(CFLAGS) -o bh bh.cpp

run: bh
	./bh -i input/nb-10.txt -o output.txt -s 10 -t 1 -d  0.005

clean:
	rm -f $(TARGET)
