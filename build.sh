#g++ main.cpp -o main -I./libs/onnxruntime/include -L./libs/onnxruntime/lib -L./lib/onnxruntime -lonnxruntime -std=c++14 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn

#g++ inference.cpp -o inference -I./libs/onnxruntime/include -L./libs/onnxruntime/lib -L./lib/onnxruntime -lonnxruntime -std=c++14 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn

g++ -g inference.cpp  -I./libs/onnxruntime/include -L./libs/onnxruntime/lib -L./lib/onnxruntime -lonnxruntime -std=c++14 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn
