#!/bin/bash
# Builds TFLite Micro (reference kernels) for the GAP8 fabric controller as libtflm.a.
# Run inside the gap8 distrobox after `source env.sh`. TREE = output of create_tflm_tree.py.
set -e
TREE=${TREE:-$HOME/wino-gap8/third_party/tflm_tree}
OUT=$PWD/build
mkdir -p $OUT
FLAGS="-march=rv32imcxgap8 -mPE=8 -mFC=1 -O2 -ffunction-sections -fdata-sections -fno-jump-tables -DTF_LITE_STATIC_MEMORY -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS -DTF_LITE_USE_GLOBAL_MAX -DTF_LITE_USE_GLOBAL_MIN -DTF_LITE_DISABLE_X86_NEON -DTF_LITE_MCU_DEBUG_LOG -I$TREE -I$TREE/third_party/flatbuffers/include -I$TREE/third_party/gemmlowp -I$TREE/third_party/ruy -I$TREE/third_party/kissfft"
CXXFLAGS="$FLAGS -std=c++17 -fno-rtti -fno-exceptions -fno-threadsafe-statics -fno-use-cxa-atexit -Wno-psabi"
objs=()
for f in $(find $TREE/tensorflow -name "*.cc" | grep -v "_test.cc\|/signal/\|benchmark\|/examples/\|test_helper\|recording\|fake_micro\|mock_micro") tflm_run.cc; do
  o=$OUT/$(echo ${f#$TREE/} | tr / _).o
  if [ ! -f $o ] || [ $f -nt $o ]; then riscv32-unknown-elf-g++ $CXXFLAGS -c $f -o $o; fi
  objs+=($o)
done
for f in $(find $TREE/tensorflow -name "*.c"); do
  o=$OUT/$(echo ${f#$TREE/} | tr / _).o
  riscv32-unknown-elf-gcc $FLAGS -c $f -o $o; objs+=($o)
done
rm -f libtflm.a
riscv32-unknown-elf-ar rcs libtflm.a "${objs[@]}"
echo "libtflm.a: ${#objs[@]} objects"
