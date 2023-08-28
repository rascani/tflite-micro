/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
<%!
    from tflite_micro.codegen import utils
%>
/* AUTOMATICALLY GENERATED DO NOT MODIFY */

#include "${header_file}"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_graph_info.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace ${model_name} {
namespace {
// TODO(rjascani): We should probably split out the OpTable to a separate file
// once we start generating for multiple models.
enum OpCode {
% for op_code in op_code_table.op_codes:
  ${op_code.enum_name},
% endfor
  kCount
};

TFLMInferenceRegistration op_table[OpCode::kCount] = {
% for op_code in op_code_table.op_codes:
    ${op_code.register_function}(),
% endfor
};

% for buffer in graph.buffers:
${buffer.generate_c_buffer_array("")}
% endfor
% for subgraph in graph.subgraphs:
${subgraph.generate_c_node_data("")}

${subgraph.generate_c_tensor_data("")}
% endfor

% if graph.needs_zero_length_int_array:
TfLiteIntArray zero_length_int_array = {};
% endif
}  // namespace

Model::Graph::Graph() {
  context_.impl_ = static_cast<void*>(this);
  context_.ReportError = nullptr;
  context_.GetTensor = nullptr;
  context_.GetEvalTensor = tflite::MicroContextGetEvalTensor;
  context_.profiler = nullptr;
  context_.GetExternalContext = nullptr;
  context_.GetScratchBuffer = nullptr;

% for subgraph in graph.subgraphs:
${subgraph.generate_c_node_init("  ")}

${subgraph.generate_c_tensor_init("  ")}
% endfor
}

void* Model::Graph::GetScratchBuffer(int buffer_idx) { return nullptr; }

TfLiteEvalTensor* Model::Graph::GetEvalTensor(int tensor_idx) {
  TfLiteEvalTensor* tensor_array = GetSubgraphTensors(current_subgraph_idx_);
  return &tensor_array[tensor_idx];
}

TfLiteStatus Model::Graph::set_external_context(
    void* external_context_payload) {
  if (external_context_payload == nullptr ||
      external_context_payload_ != nullptr) {
    MicroPrintf(
        "Attempting to set external context to %x but it was %x already",
        external_context_payload, external_context_payload_);
    return kTfLiteError;
  }

  external_context_payload_ = external_context_payload;
  return kTfLiteOk;
}

void* Model::Graph::external_context() { return external_context_payload_; }

tflite::MicroGraphInfo& Model::Graph::graph_info() { return *this; }

TfLiteStatus Model::Graph::InvokeSubgraph(int subgraph_idx) {
  int previous_subgraph_idx = current_subgraph_idx_;
  TfLiteStatus status = kTfLiteError;
  switch (subgraph_idx) {
%for subgraph in graph.subgraphs:
    case ${subgraph.index}:
      status = InvokeSubgraph${subgraph.index}();
      break;
%endfor
    default:
      break;
  }
  current_subgraph_idx_ = previous_subgraph_idx;
  return status;
}

size_t Model::Graph::NumSubgraphInputs(int subgraph_idx) {
  switch (subgraph_idx) {
%for subgraph in graph.subgraphs:
    case ${subgraph.index}:
      return ${len(subgraph.inputs)};
%endfor
  }
  return 0;
}

TfLiteEvalTensor* Model::Graph::GetSubgraphInput(int subgraph_idx,
                                                 int input_idx) {
  int tensor_idx = GetTensorInputIndex(subgraph_idx, input_idx);
  return &GetSubgraphTensors(subgraph_idx)[tensor_idx];
}

size_t Model::Graph::NumSubgraphOutputs(int subgraph_idx) {
  switch (subgraph_idx) {
%for subgraph in graph.subgraphs:
    case ${subgraph.index}:
      return ${len(subgraph.outputs)};
%endfor
  }
  return 0;
}

TfLiteEvalTensor* Model::Graph::GetSubgraphOutput(int subgraph_idx,
                                                  int output_idx) {
  int tensor_idx = GetTensorOutputIndex(subgraph_idx, output_idx);
  return &GetSubgraphTensors(subgraph_idx)[tensor_idx];
}

tflite::MicroResourceVariables* Model::Graph::GetResourceVariables() {
  // TODO(rjascani): Handle MicroResourceVariables
  return nullptr;
}

TfLiteEvalTensor* Model::Graph::GetSubgraphTensors(int subgraph_idx) {
  switch (subgraph_idx) {
%for subgraph in graph.subgraphs:
    case ${subgraph.index}:
      return ${subgraph.tensors_array};
%endfor
  }
  return nullptr;
}

int Model::Graph::GetTensorInputIndex(int subgraph_idx, int input_idx) {
  switch (subgraph_idx) {
%for subgraph in graph.subgraphs:
    case ${subgraph.index}: {
${utils.generate_c_int_array("      ", "kInputs", subgraph.inputs)}
      return kInputs[input_idx];
    }
%endfor
  }
  return 0;
}

int Model::Graph::GetTensorOutputIndex(int subgraph_idx, int output_idx) {
  switch (subgraph_idx) {
%for subgraph in graph.subgraphs:
    case ${subgraph.index}: {
${utils.generate_c_int_array("      ", "kOutputs", subgraph.outputs)}
      return kOutputs[output_idx];
    }
%endfor
  }
  return 0;
}

% for subgraph in graph.subgraphs:
TfLiteStatus Model::Graph::InvokeSubgraph${subgraph.index}() {
${subgraph.generate_c_invoke("  ")}
  return kTfLiteOk;
}
% endfor

TfLiteStatus Model::Invoke() { return graph_.InvokeSubgraph(0); }

}  // namespace ${model_name}
