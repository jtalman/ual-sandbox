set title 'E_BM_M1.0_sl4.sxf'
set xlabel 'Longitudinal Position, S'
set ylabel 'Beta Y'
set xrange [0:16.5]
set yrange [261.6:263.6]
#set autoscale y 

p 'betaFunctions' u 2:4 w l
