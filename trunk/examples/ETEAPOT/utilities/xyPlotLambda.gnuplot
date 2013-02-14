set size ratio 1
set xrange [-25:25]
set xlabel "x[m]"
set yrange [-25:25]
set ylabel "y[m]"
p "xyPlotLambda.out" u 1:2 w l
