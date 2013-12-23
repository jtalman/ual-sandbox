# gnuplot> load "betax.gnuplot"

 set xlabel "longitudinal position, s"
 set ylabel "{/Symbol b}_x"
 set xrange [0.0:16.6]
# set yrange [36.05:36.25]
 set grid
 set title 'E\_BM\_Z\_sl4, m=0'
 # set terminal postscript eps enhanced 20 
 # set output "E\_BM_Z\_sl4_beta_x.eps"
 plot "betaFunctions" using 2:3 with lines
