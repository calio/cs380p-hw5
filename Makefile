# Makefile for MPI program

CC=mpic++              # Compiler to use
CFLAGS=-Wall -O2       # Compiler flags, e.g., optimization and warning flags
TARGET=nbody # Target executable name
SRC=main.cpp  # Source file

all: $(TARGET)

#$(TARGET): $(SRC)
#	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

nbody: main.cpp BarnesHut.h
	$(CC) $(CFLAGS) -o nbody main.cpp

run: nbody
	mpirun -np 2 ./nbody -i input/nb-10.txt -o output.txt -s 10 -t 1 -d  0.005

#bh: bh.cpp
#	g++ $(CFLAGS) -o bh bh.cpp

run_bh: bh
	./bh -i input/nb-10.txt -o output.txt -s 10 -t 1 -d  0.005

clean:
	rm -f $(TARGET)
