#include <iostream>
#include <onnxruntime_cxx_api.h>

#include "Helpers.cpp"

using namespace std;

int main() {
  Ort::Env env;
  Ort::RunOptions runOptions;
  Ort::Session session(nullptr);

  constexpr int64_t numChannels = 3;
  constexpr int64_t width = 224;
  constexpr int64_t height = 224;
  constexpr int64_t numClasses = 3;
  constexpr int64_t numInputElements = numChannels * height * width;

  const string imageFile = "/home/ctyler/personal/cpp-inference/assets/dog.png";
  const string labelFile =
      "/home/ctyler/personal/cpp-inference/assets/imagenet_classes.txt";
  auto modelPath =
      L"/home/ctyler/personal/cpp-inference/assets/resnet50v2.onnx";

  // load labels
  vector<string> labels = loadLabels(labelFile);
  if (labels.empty()) {
    cout << "Failed to laod labels: " << labelFile << endl;
    return 1;
  };

  // load image
  const vector<float> imageVec = loadImage(imageFile);
  if (imageVec.empty()) {
    cout << "Failed to load image: " << imageFile << endl;
    return 1;
  }

  if (imageVec.size() != numInputElements) {
    cout << "Invalid image format. Must be 224x224 RGB image." << endl;
    return 1;
  }

  // Use CUDA GPU
  Ort::SessionOptions ort_session_options;

  OrtCUDAProviderOptions options;
  options.device_id = 0;
  // options.arena_extend_strategy = 0;
  // options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
  // options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  // options.do_copy_in_default_stream = 1;

  OrtSessionOptionsAppendExecutionProvider_CUDA(ort_session_options,
                                                options.device_id);

  // create session
  session = Ort::Session(env, modelPath, ort_session_options);

  // use CPU
  // session = Ort::Session(env, modelPath, Ort::SessionOptions{nullptr});

  // define shape
  const array<int64_t, 4> inputShape = {1, numChannels, height, width};
  const array<int64_t, 2> outputShape = {1, numClasses};

  // define array
  array<float, numInputElements> input;
  array<float, numClasses> results;

  // define tensor
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto inputTensor =
      Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(),
                                      inputShape.data(), inputShape.size());
  auto outputTensor = Ort::Value::CreateTensor<float>(
      memory_info, results.data(), results.size(), outputShape.data(),
      outputShape.size());

  // copy image data to input array
  copy(imageVec.begin(), imageVec.end(), input.begin());

  // define name
  Ort::AllocatorWithDefaultOptions ort_alloc;
  char *inputName = session.GetInputName(0, ort_alloc);
  char *outputName = session.GetOutputName(0, ort_alloc);
  const array<const char *, 1> inputNames = {inputName};
  const array<const char *, 1> outputNames = {outputName};
  ort_alloc.Free(inputName);
  ort_alloc.Free(outputName);

  // run inference
  try {
    session.Run(runOptions, inputNames.data(), &inputTensor, 1,
                outputNames.data(), &outputTensor, 1);
  } catch (Ort::Exception &e) {
    cout << e.what() << endl;
  }

  // sort results
  vector<pair<size_t, float>> indexValuePairs;
  for (size_t i = 0; i < results.size(); ++i) {
    indexValuePairs.emplace_back(i, results[i]);
  }
  sort(
      indexValuePairs.begin(), indexValuePairs.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.second > rhs.second; });

  for (size_t i = 0; i < 5; ++i) {
    const auto &result = indexValuePairs[i];
    cout << i + 1 << ": " << labels[result.first] << " " << result.second
         << endl;
  }
}
