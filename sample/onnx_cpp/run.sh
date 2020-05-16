PWD=`pwd`
echo $PWD
LD_LIBRARY_PATH=$PWD/../../dep/onnxruntime-linux-x64-1.2.0/lib ./a.out
