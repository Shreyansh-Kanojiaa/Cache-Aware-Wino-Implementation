# GAP8 layer study: direct vs Winograd 3x3 convs, float32 (software) and int8, FC vs cluster.
# Inside the gap8 distrobox, after `source env.sh` from the UROP gap8 project:
#   make clean all run platform=gvsoc      (simulator)
#   make clean all run platform=board      (GAPuino over JTAG)
APP        = layerbench
APP_SRCS   = main.c
APP_CFLAGS = -O2 $(EXTRA)
include $(RULES_DIR)/pmsis_rules.mk
