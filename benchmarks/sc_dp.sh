# TEMP: median ms of direct_packed shapes: "k s Cin Cout H" per line on stdin; extra harness args in $@
while read k s ci co h; do
  rm -f /tmp/dp.csv
  build/bench_harness --variant direct_packed --k $k --stride $s --Cin $ci --Cout $co --H $h --W $h --T 1 --warmup 2 --runs 7 --out /tmp/dp.csv "$@"
  echo "k$k s$s $ci->$co H$h $* : $(cut -d, -f13 /tmp/dp.csv | sort -n | sed -n 4p) ms"
done
