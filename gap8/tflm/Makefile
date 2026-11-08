# TFLite Micro on the GAP8 fabric controller. Build libtflm.a first (./build_lib.sh), then
#   make clean all run platform=board
APP          = tflm_resnet8
APP_SRCS     = main.c stubs.c
APP_CFLAGS   = -O2
APP_LDFLAGS  = $(CURDIR)/libtflm.a -lstdc++ -lsupc++ -lm
include $(RULES_DIR)/pmsis_rules.mk
