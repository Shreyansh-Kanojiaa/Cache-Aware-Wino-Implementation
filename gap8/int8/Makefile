# int8 engine on the GAPuino. Inside the gap8 distrobox, after `source env.sh` in the UROP gap8 project:
#   make clean all run platform=board MODEL=resnet8
# Needs models/net8_MODEL.h and models/MODEL.{w8,in8,ref8} from gen8.py.
MODEL       ?= resnet8
APP          = int8_$(MODEL)
APP_SRCS     = main8.c
APP_CFLAGS   = -O3 $(EXTRA) -DNET='"models/net8_$(MODEL).h"' -DMODEL='"$(MODEL)"'
MODEL_FILES  = models/$(MODEL).w8 models/$(MODEL).u8 models/$(MODEL).in8 models/$(MODEL).ref8
override config_args += $(foreach f, $(MODEL_FILES), --config-opt=flash/content/partitions/readfs/files=$(f))
include $(RULES_DIR)/pmsis_rules.mk
