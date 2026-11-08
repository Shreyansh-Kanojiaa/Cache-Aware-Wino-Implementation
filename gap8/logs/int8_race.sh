#!/bin/bash
# The int8 race on the GAPuino at 100 MHz (each app sets the clock as DORY's main does)
B=~/wino-gap8/gap8/board_run.sh; L=~/wino-gap8/gap8/logs
TIMEOUT=900  $B ~/wino-gap8/gap8/int8 $L/race_ours_resnet8.log MODEL=resnet8
TIMEOUT=900  $B ~/wino-gap8/third_party/dory_apps/resnet8_perf $L/race_dory_resnet8.log CORE=8
TIMEOUT=900  $B ~/wino-gap8/gap8/tflm $L/race_tflm_resnet8.log
TIMEOUT=1800 $B ~/wino-gap8/third_party/dory_apps/mv2_perf $L/race_dory_mv2.log CORE=8
TIMEOUT=1800 $B ~/wino-gap8/gap8/int8 $L/race_ours_mv2.log MODEL=mv2
echo race-done
