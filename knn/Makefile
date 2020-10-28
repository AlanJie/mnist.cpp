CC=g++
INCLUDE_DIR := $(MNIST_ML_ROOT)/knn/include
SRC := $(MNIST_ML_ROOT)/knn/src
CFLAGS := -std=c++11 -DEUCLID

all : main

main : $(SRC)/knn.cpp
	$(CC) $(CFLAGS) $(SRC)/knn.cpp -o main -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) -I$(MNIST_ML_ROOT)/include -ldata

clean :
	rm main
