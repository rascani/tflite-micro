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

/* AUTOMATICALLY GENERATED DO NOT MODIFY */

#pragma once

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_graph_info.h"

namespace ${model_name} {

class Model {
 public:
  Model() = default;

  TfLiteStatus Invoke();

 private:
  class Graph : public tflite::MicroContext, public tflite::MicroGraphInfo {
   public:
    Graph();

    // MicroContext API
    void* GetScratchBuffer(int buffer_idx) override;

    TfLiteEvalTensor* GetEvalTensor(int tensor_idx) override;

    TfLiteStatus set_external_context(void* external_context_payload) override;

    void* external_context() override;

    tflite::MicroGraphInfo& graph_info() override;

    void* AllocatePersistentBuffer(size_t) override { return nullptr; }

    TfLiteStatus RequestScratchBufferInArena(size_t, int*) override {
      return kTfLiteError;
    }

    TfLiteTensor* AllocateTempTfLiteTensor(int) override { return nullptr; }

    void DeallocateTempTfLiteTensor(TfLiteTensor*) override {}

    uint8_t* AllocateTempBuffer(size_t, size_t) override { return nullptr; }

    void DeallocateTempBuffer(uint8_t*) override {}

    // MicroGraphInfo API
    TfLiteStatus InvokeSubgraph(int subgraph_idx) override;

    size_t NumSubgraphInputs(int subgraph_idx) override;

    TfLiteEvalTensor* GetSubgraphInput(int subgraph_idx,
                                       int input_idx) override;

    size_t NumSubgraphOutputs(int subgraph_idx) override;

    TfLiteEvalTensor* GetSubgraphOutput(int subgraph_idx,
                                        int output_idx) override;

    int NumSubgraphs() override { return ${len(graph.subgraphs)}; }

    tflite::MicroResourceVariables* GetResourceVariables() override;

   private:
    TfLiteEvalTensor* GetSubgraphTensors(int subgraph_idx);
    int GetTensorInputIndex(int subgraph_idx, int input_idx);
    int GetTensorOutputIndex(int subgraph_idx, int output_idx);

% for subgraph_idx in range(len(graph.subgraphs)):
    TfLiteStatus InvokeSubgraph${subgraph_idx}();
% endfor

    TfLiteContext context_ = {};
    int current_subgraph_idx_ = 0;
    void* external_context_payload_;
% for subgraph in graph.subgraphs:
    TfLiteNode ${subgraph.nodes_array}[${len(subgraph.operators)}] = {};
% endfor
% for subgraph in graph.subgraphs:
    TfLiteEvalTensor ${subgraph.tensors_array}[${len(subgraph.tensors)}] = {};
% endfor
    TF_LITE_REMOVE_VIRTUAL_DELETE
  };

  Graph graph_ = {};
};

}  // namespace ${model_name}
