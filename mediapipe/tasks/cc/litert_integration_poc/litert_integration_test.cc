// Proof-of-concept verifying that @litert targets can be built and linked
// alongside a real mediapipe package (mediapipe/framework/port:logging) in
// the same binary. This is scoped intentionally: it must never be linked
// into any target that also pulls in @org_tensorflow//tensorflow/lite/...,
// since LiteRT's tflite:: namespace is the same one org_tensorflow's TFLite
// uses - linking both risks duplicate-symbol/ODR problems.
//
// To add a new @litert target to this smoke test: add its check function
// below, add it to kChecks, and add its BUILD dep in the BUILD file.
//
// Build with:
//   bazel build //mediapipe/tasks/cc/litert_integration_poc:litert_integration_test \
//       --noapple_generate_dsym --features=-layering_check

#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "mediapipe/framework/port/logging.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tflite/model_builder.h"

namespace {

struct Check {
  const char* name;
  bool (*run)();
};

// Deliberately invalid flatbuffer data - just enough to prove the model
// verification/loading code actually runs (not just links) and correctly
// rejects garbage input.
constexpr char kInvalidModelData[] = "not a flatbuffer";

// @litert//tflite:model_builder
bool CheckTfliteModelBuilder() {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
          kInvalidModelData, sizeof(kInvalidModelData) - 1);
  return model == nullptr;
}

// @litert//litert/cc:litert_model
bool CheckLitertModel() {
  litert::BufferRef<uint8_t> buffer(kInvalidModelData,
                                     sizeof(kInvalidModelData) - 1);
  litert::Expected<litert::Model> model =
      litert::Model::CreateFromBuffer(buffer);
  return !model.HasValue();
}

// @litert//litert/cc:litert_options
bool CheckLitertOptions() {
  litert::Expected<litert::Options> options = litert::Options::Create();
  return options.HasValue();
}

// @litert//litert/cc:litert_compiled_model
//
// litert::CompiledModel::Create() needs a real, valid litert::Model, which
// needs a real .tflite file - out of scope for this header/link smoke test.
// This just proves litert::Environment::Create() (a real prerequisite
// CompiledModel needs) builds, links, and runs.
bool CheckLitertCompiledModel() {
  litert::Expected<litert::Environment> env =
      litert::Environment::Create({});
  return env.HasValue();
}

// @litert//litert/c:litert_model_types
//
// Pure C enum, no runtime logic to exercise - just confirms the header
// resolves and the constant has the value the .h file documents.
bool CheckLitertModelTypes() {
  return kLiteRtElementTypeFloat32 == 1;
}

// @litert//litert/cc:litert_any
bool CheckLitertAny() {
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeInt;
  litert_any.int_value = 42;
  litert::LiteRtVariant value = litert::ToStdAny(litert_any);
  return std::holds_alternative<int64_t>(value) &&
         std::get<int64_t>(value) == 42;
}

// @litert//litert/cc:litert_tensor_buffer
bool CheckLitertTensorBuffer() {
  litert::Expected<litert::Environment> env = litert::Environment::Create({});
  if (!env.HasValue()) return false;
  litert::RankedTensorType tensor_type(
      litert::ElementType::Float32,
      litert::Layout(litert::Dimensions{1, 3, 224, 224}));
  litert::Expected<litert::TensorBuffer> buffer =
      litert::TensorBuffer::CreateManaged(
          *env, litert::TensorBufferType::kHostMemory, tensor_type,
          /*buffer_size=*/1 * 3 * 224 * 224 * sizeof(float));
  return buffer.HasValue();
}

constexpr Check kChecks[] = {
    {"tflite:model_builder", &CheckTfliteModelBuilder},
    {"litert/cc:litert_model", &CheckLitertModel},
    {"litert/cc:litert_options", &CheckLitertOptions},
    {"litert/cc:litert_compiled_model (Environment only)",
     &CheckLitertCompiledModel},
    {"litert/c:litert_model_types", &CheckLitertModelTypes},
    {"litert/cc:litert_any", &CheckLitertAny},
    {"litert/cc:litert_tensor_buffer", &CheckLitertTensorBuffer},
};

}  // namespace

int main(int argc, char** argv) {
  LOG(INFO) << "Verifying @litert targets link and run alongside "
               "mediapipe/framework/port:logging...";
  bool all_passed = true;
  for (const Check& check : kChecks) {
    bool passed = check.run();
    LOG(INFO) << "[" << (passed ? "PASS" : "FAIL") << "] " << check.name;
    all_passed &= passed;
  }
  return all_passed ? 0 : 1;
}
