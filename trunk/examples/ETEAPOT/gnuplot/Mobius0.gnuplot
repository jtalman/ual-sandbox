# gnuplot> load "gnuplot/Mobius0.gnuplot"

set grid
set pointsize 3

# set terminal x11 1
set xlabel "x_{max} [m]"
set ylabel "s_x"
set title 'Mobius0-xy-sx, m=-1.2, 63470 turns'
set terminal postscript eps enhanced 20 
set output "eps/Mobius0-xy-sx.eps"
plot "gnuplot/Mobius0.xy-data" u 2:4:(0.0003) w yerrorbars, -2.0*x*x

# set terminal x11 3
set xlabel "de_0"
set ylabel "s_x"
set title 'Mobius0-de-sx, m=-1.2, 63470 turns'
set terminal postscript eps enhanced 20 
set output "eps/Mobius0-de-sx.eps"
plot "gnuplot/Mobius0.de-data" u 5:4 w p, "gnuplot/Mobius0.de-data" u 5:4 smooth csplines, 8600000*x*x





