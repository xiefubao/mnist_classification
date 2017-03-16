g++-5 -o read  train.cpp `pkg-config opencv --cflags --libs`
./read
rm read