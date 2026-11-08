# GAP8 whole-model runner. Inside the gap8 distrobox, after `source env.sh` in the UROP gap8 project:
#   make clean all run platform=board MODEL=resnet8
# Needs models/MODEL.{bin,input,ref} (export_gap8.py on the Pi), then gen.py for net_MODEL.h and models/MODEL.u.
MODEL       ?= resnet8
APP          = runner_$(MODEL)
APP_SRCS     = main.c
APP_CFLAGS   = -O2 $(EXTRA) -DNET='"net_$(MODEL).h"' -DMODEL='"$(MODEL)"'
# Listed once ourselves: READFS_FILES is added by both pmsis_rules.mk and pulp_rules.mk in
# gap_sdk 4.22, which stores every file twice and doubles the flash used.
MODEL_FILES  = models/$(MODEL).bin models/$(MODEL).u models/$(MODEL).input models/$(MODEL).ref
override config_args += $(foreach f, $(MODEL_FILES), --config-opt=flash/content/partitions/readfs/files=$(f))
include $(RULES_DIR)/pmsis_rules.mk
