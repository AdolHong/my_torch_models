#include <onnxruntime_cxx_api.h>
// 参考 github: microsoft/onnxruntime: onnxruntime/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp, 
// https://github.com/microsoft/onnxruntime/blob/7494500221c1de4beee9658feb38a3d1d17737a5/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
#include <string>
#include <vector>
#include <memory>
#include <iostream>
using namespace std;

int main(){
    std::string fp_model = "../multiclass_iris/softmax.onnx";
    int num_threads = 5;
    int num_features = 4;
    int num_class = 3;
    
    std::shared_ptr<Ort::Session> session;
    std::shared_ptr<Ort::Env> env;
    
    // 初始化环境， 每个进程一个env
    env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "production"); // 环境名称

    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session = std::make_shared<Ort::Session>(*env, fp_model.c_str(), session_options);

    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;
  
    // print number of model input nodes
    size_t num_input_nodes = session->GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                           // Otherwise need vector<vector<>>

    printf("Number of inputs = %zu\n", num_input_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
        // print input node names
        char* input_name = session->GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;
  
        // print input node types
        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
          printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }

    // std::vector<const char*> input_node_names = {"input"};
    // std::vector<int64_t> input_node_dims = {1, 4};
    std::vector<float> input_tensor_values = {6.2, 3.4, 5.4, 2.3};
    std::vector<const char*> output_node_names = {"output"};
      // create input tensor object from data values
      auto memory_info =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // 参考 github: microsoft/onnxruntime: onnxruntime/onnxruntime/core/framework/onnxruntime_typeinfo.cc
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, input_tensor_values.data(), num_features,
          input_node_dims.data(), 2); // 此处的2是输入的dim

      // std::vector<Value> Run(const RunOptions& run_options, const char*
      // const*input_names, const Value* input_values, size_t input_count, const
      // char* const* output_names, size_t output_count);
      auto output_tensors =
          session->Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                       &input_tensor, 1, output_node_names.data(), 1);

      float* floatarr = output_tensors.front().GetTensorMutableData<float>();
      for (unsigned i = 0; i < num_class; ++i) {
          std::cout<<floatarr[i]<<" ";
      }
      std::cout<<std::endl;

    return 0;
}
