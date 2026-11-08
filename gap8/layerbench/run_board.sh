#!/bin/bash
# Runs the layer study on the GAPuino (JTAG): ./run_board.sh [make variables, e.g. EXTRA=-DQUICK]
cd /home/shreyansh/Projects/UROP/gap8
timeout ${TIMEOUT:-1500} distrobox enter gap8 -- bash -c "source env.sh >/dev/null 2>&1; cd ~/wino-gap8/gap8/layerbench && make clean all run platform=board $*"
echo "exit $?"
