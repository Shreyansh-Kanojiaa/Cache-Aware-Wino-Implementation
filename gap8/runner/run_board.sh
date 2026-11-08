#!/bin/bash
# ./run_board.sh MODEL: build, flash and run on the GAPuino; log in logs/MODEL.log
cd /home/shreyansh/Projects/UROP/gap8
mkdir -p ~/wino-gap8/gap8/runner/logs
timeout ${TIMEOUT:-7200} distrobox enter gap8 -- bash -c "source env.sh >/dev/null 2>&1; cd ~/wino-gap8/gap8/runner && make clean all run platform=board MODEL=$1" > ~/wino-gap8/gap8/runner/logs/$1.log 2>&1
echo "exit $?" >> ~/wino-gap8/gap8/runner/logs/$1.log
