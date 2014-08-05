# gnuplot> load "gnuplot/betay.gnuplot"

 set xlabel "longitudinal position, s"
 set ylabel "{/Symbol b}_y"
# set xrange [0.0:21.451]
# set yrange [263.4:263.5]
 set grid
 set title 'E\_pEDM.rtr-betay, m=-1.2'
 set terminal postscript eps enhanced 20 
 set output "eps/E\_pEDM.rtr-betay.eps"
 plot "E_pEDM-rtr1-betafunctions" using 2:4 with lines
