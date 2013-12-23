# gnuplot> load "betay.gnuplot"

 set xlabel "longitudinal position, s"
 set ylabel "{/Symbol b}_y"
 set xrange [0.0:16.6]
# set yrange [263.4:263.5]
 set grid
 set title 'E\_BM\_Z\_sl4, m=0'
 # set terminal postscript eps enhanced 20 
 # set output "E\_BM_Z\_sl4_beta_y.eps"
 plot "betaFunctions" using 2:4 with lines
