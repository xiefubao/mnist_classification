g++-5 -o read  get_mnist.cpp `pkg-config opencv --cflags --libs`
./read
rm read
