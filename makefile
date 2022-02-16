
CC=g++
NVCC=nvcc

INCDIRS=-I/usr/local/cuda-11.5/include

CPP= src/algorithms/SynC.cpp src/utils/CPU_math.cpp

CU= src/algorithms/GPU_SynC.cu src/utils/GPU_utils.cu

CUFLAGS=-arch=sm_75 --ptxas-options=-v

debug: src/main.cpp
	$(NVCC) -o bin/debug/main src/main.cpp $(CPP) $(CU) $(INCDIRS) $(CUFLAGS) -g -G

run_debug:
	python generate.py $(n) $(d) $(cl)
	./bin/debug/main $(n) $(d) $(cl)

release: src/main.cpp
	$(NVCC) -o bin/release/main src/main.cpp $(CPP) $(CU) $(INCDIRS) $(CUFLAGS) -O3

run_release:
	python generate.py $(n) $(d) $(cl)
	./bin/release/main $(n) $(d) $(cl) $(v)
