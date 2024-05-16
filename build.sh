#g++ main.cpp -o main -v -Wall -I/home/ctyler/personal/cpp-inference/libs/onnxruntime2/include -L/home/ctyler/personal/cpp-inference/libs/onnxruntime2/lib -L/home/ctyler/personal/cpp-inference/libs -lonnxruntime2 -static-libstdc++

g++ main.cpp -o main -I./libs/onnxruntime/include -L./libs/onnxruntime/lib -L./lib/onnxruntime -lonnxruntime -std=c++14 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn
