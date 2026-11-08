// C interface to TFLite Micro for the GAP8 runner: load a model, run it on int8 inputs.
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace {
// placement new, not globals with constructors: no reliance on static initialisation
typedef tflite::MicroMutableOpResolver<8> Resolver;
tflite::MicroInterpreter *interp;
alignas(16) uint8_t interp_mem[sizeof(tflite::MicroInterpreter)];
alignas(16) uint8_t resolver_mem[sizeof(Resolver)];
}

extern "C" int tflm_init(const void *model, uint8_t *arena, int arena_bytes) {
    const tflite::Model *m = tflite::GetModel(model);
    if (m->version() != TFLITE_SCHEMA_VERSION) return -1;
    Resolver *r = new (resolver_mem) Resolver();
    r->AddConv2D(); r->AddDepthwiseConv2D(); r->AddFullyConnected(); r->AddAdd();
    r->AddAveragePool2D(); r->AddReshape(); r->AddSoftmax(); r->AddMaxPool2D();
    interp = new (interp_mem) tflite::MicroInterpreter(m, *r, arena, arena_bytes);
    if (interp->AllocateTensors() != kTfLiteOk) return -2;
    return (int)interp->arena_used_bytes();
}
extern "C" int8_t *tflm_input(int *n, float *scale, int *zp) {
    TfLiteTensor *t = interp->input(0);
    *n = t->bytes; *scale = t->params.scale; *zp = t->params.zero_point;
    return t->data.int8;
}
extern "C" int8_t *tflm_output(int *n) { TfLiteTensor *t = interp->output(0); *n = t->bytes; return t->data.int8; }
extern "C" int tflm_invoke(void) { return interp->Invoke() == kTfLiteOk ? 0 : -1; }
