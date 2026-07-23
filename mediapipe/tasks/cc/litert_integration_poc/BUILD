# Proof-of-concept only: verifies @litert builds, links, and runs alongside
# a real mediapipe package in the same binary. Not part of any production
# target. See litert_integration_test.cc for the ODR-safety caveat on
# combining @litert with @org_tensorflow//tensorflow/lite/... targets.
#
# Build with:
#   bazel build //mediapipe/tasks/cc/litert_integration_poc:litert_integration_test \
#       --noapple_generate_dsym --features=-layering_check

cc_binary(
    name = "litert_integration_test",
    srcs = ["litert_integration_test.cc"],
    deps = [
        "//mediapipe/framework/port:logging",
        "@litert//litert/c:litert_model_types",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_any",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//tflite:model_builder",
    ],
)
