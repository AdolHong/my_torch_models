PWD=`pwd`
echo $PWD
g++ -o a.out demo.cc -L $PWD/../../deps/onnxruntime-linux-x64-1.2.0/lib -lonnxruntime -I $PWD/../../deps/onnxruntime-linux-x64-1.2.0/include -std=c++11
