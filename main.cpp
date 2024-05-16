#include "libs/onnxruntime2/include/onnxruntime_c_api.h"
#include "libs/onnxruntime2/include/onnxruntime_cxx_api.h"

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

template <typename T> T vectorProduct(const std::vector<T> &v) {
  return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

std::vector<std::string> readLabels(std::string &labelFilepath) {
  std::vector<std::string> labels;
  std::string line;
  std::ifstream fp(labelFilepath);
  while (std::getline(fp, line)) {
    labels.push_back(line);
  }
  return labels;
}

int main() {
  const int64_t batchSize = 2;
  bool useCUDA{true};

  std::string instanceName{"image-classification-inference"};
  std::string modelFilePath{"./data/resnet18-v1-7.onnx"};
  std::string imageFilePath{"./data/birb.jpg"};
  std::string labelFilePath{"./data/synset.txt"};

  std::vector<std::string> labels{readLabels(labelFilePath)};

  Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
               instanceName.c_str());
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetInterOpNumThreads(1);
  if (useCUDA) {
    OrtCUDAProviderOptions cuda_options{};
    sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
  }

  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Ort::Session session(env, modelFilePath.c_str(), sessionOptions);

  Ort::AllocatorWithDefaultOptions allocator;

  size_t numInputNodes = session.GetInputCount();
  size_t numOutputNodes = session.GetOutputCount();

  const char *inputName = session.GetInputNameAllocated(0, allocator).get();

  Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
  auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

  ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

  std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
  if (inputDims.at(0) == -1) {
    std::cout << "Got dynamic batch size. Setting input batch size to "
              << batchSize << "." << std::endl;
    inputDims.at(0) = batchSize;
  }

  const char *outputName = session.GetOutputNameAllocated(0, allocator).get();

  Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
  auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

  ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

  std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
  if (outputDims.at(0) == -1) {
    std::cout << "Got dynamic batch size. Setting output batch size to "
              << batchSize << "." << std::endl;
    outputDims.at(0) = batchSize;
  }

  std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
  std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
  std::cout << "Input Name: " << inputName << std::endl;
  std::cout << "Input Type: " << inputType << std::endl;
  // std::cout << "Input Dimensions: " << inputDims << std::endl;
  std::cout << "Output Name: " << outputName << std::endl;
  std::cout << "Output Type: " << outputType << std::endl;
  // std::cout << "Output Dimensions: " << outputDims << std::endl;

  cv::Mat imageBGR = cv::imread(imageFilePath, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
  cv::resize(imageBGR, resizedImageBGR,
             cv::Size(inputDims.at(3), inputDims.at(2)),
             cv::InterpolationFlags::INTER_CUBIC);
  cv::cvtColor(resizedImageBGR, resizedImageRGB,
               cv::ColorConversionCodes::COLOR_BGR2RGB);
  resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

  cv::Mat channels[3];
  cv::split(resizedImage, channels);

  channels[0] = (channels[0] - 0.485) / 0.229;
  channels[1] = (channels[1] - 0.456) / 0.224;
  channels[2] = (channels[2] - 0.406) / 0.225;
  cv::merge(channels, 3, resizedImage);
  cv::dnn::blobFromImage(resizedImage, preprocessedImage);

  size_t inputTensorSize = vectorProduct(inputDims);
  std::vector<float> inputTensorValues(inputTensorSize);

  for (int64_t i = 0; i < batchSize; ++i) {
    std::copy(preprocessedImage.begin<float>(), preprocessedImage.end<float>(),
              inputTensorValues.begin() + i * inputTensorSize / batchSize);
  }

  size_t outputTensorSize = vectorProduct(outputDims);
  assert(("Output tensor size should equal to the label set size.",
          labels.size() * batchSize == outputTensorSize));
  std::vector<float> outputTensorValues(outputTensorSize);

  std::vector<const char *> inputNames{inputName};
  std::vector<const char *> outputNames{outputName};
  std::vector<Ort::Value> inputTensors;
  std::vector<Ort::Value> outputTensors;

  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  inputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
      inputDims.size()));
  outputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo, outputTensorValues.data(), outputTensorSize,
      outputDims.data(), outputDims.size()));

  session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
              1 /*Number of inputs*/, outputNames.data(), outputTensors.data(),
              1 /*Number of outputs*/);

  std::vector<int> predIds(batchSize, 0);
  std::vector<std::string> predLabels(batchSize);
  std::vector<float> confidences(batchSize, 0.0f);
  for (int64_t b = 0; b < batchSize; ++b) {
    float activation = 0;
    float maxActivation = std::numeric_limits<float>::lowest();
    float expSum = 0;
    for (int i = 0; i < labels.size(); i++) {
      activation = outputTensorValues.at(i + b * labels.size());
      expSum += std::exp(activation);
      if (activation > maxActivation) {
        predIds.at(b) = i;
        maxActivation = activation;
      }
    }
    predLabels.at(b) = labels.at(predIds.at(b));
    confidences.at(b) = std::exp(maxActivation) / expSum;
  }
  for (int64_t b = 0; b < batchSize; ++b) {
    assert(("Output predictions should all be identical.",
            predIds.at(b) == predIds.at(0)));
  }
  // All the predictions should be the same
  // because the input images are just copies of each other.

  std::cout << "Predicted Label ID: " << predIds.at(0) << std::endl;
  std::cout << "Predicted Label: " << predLabels.at(0) << std::endl;
  std::cout << "Uncalibrated Confidence: " << confidences.at(0) << std::endl;

  // Measure latency
  int numTests{100};
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  for (int i = 0; i < numTests; i++) {
    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Minimum Inference Latency: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     begin)
                       .count() /
                   static_cast<float>(numTests)
            << " ms" << std::endl;

  return 0;
}
