#!/bin/bash
# ./board_run.sh APP_DIR LOG [make variables]: build, flash and run an app on the GAPuino
cd /home/shreyansh/Projects/UROP/gap8
d=$1; log=$2; shift 2
timeout ${TIMEOUT:-3600} distrobox enter gap8 -- bash -c "source env.sh >/dev/null 2>&1; cd $d && make clean all run platform=board $*" > $log 2>&1
echo "exit $?" >> $log
pkill -x gap8-openocd   # a timed-out run leaves it holding the JTAG adapter
