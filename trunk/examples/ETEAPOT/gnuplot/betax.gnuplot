# gnuplot> load "gnuplot/betax.gnuplot"

 set xlabel "longitudinal position, s"
 set ylabel "{/Symbol b}_x"
# set xrange [0.0:21.451]
# set yrange [36.05:36.25]
 set grid
 set title 'E\_pEDM.rtr-betax, m=-1.2'
 set terminal postscript eps enhanced 20 
 set output "eps/E\_pEDM.rtr-betax.eps"
 plot "E_pEDM-rtr1-betafunctions" using 2:3 with lines
