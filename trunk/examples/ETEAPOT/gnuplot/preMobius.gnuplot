# gnuplot> load "gnuplot/preMobius.gnuplot"

set grid
set pointsize 3

# set terminal x11 1
set xlabel "x_{max} [m]"
set ylabel "s_x"
set title 'preMobius-x-sx, m=-1.2, 63470 turns'
set terminal postscript eps enhanced 20 
set output "eps/preMobius-x-sx.eps"
plot "gnuplot/preMobius.x-data" u 2:4 w p, "gnuplot/preMobius.x-data" u 2:4 smooth csplines, -240*x*x

# set terminal x11 2
set xlabel "y_{max} [m]"
set ylabel "s_x"
set title 'preMobius-y-sx, m=-1.2, 63470 turns'
set terminal postscript eps enhanced 20 
set output "eps/preMobius-y-sx.eps"
plot "gnuplot/preMobius.y-data" u 3:4 w p, "gnuplot/preMobius.y-data" u 3:4 smooth csplines, 104*x*x

# set terminal x11 3
set xlabel "de_0"
set ylabel "s_x"
set title 'preMobius-de-sx, m=-1.2, 63470 turns'
set terminal postscript eps enhanced 20 
set output "eps/preMobius-de-sx.eps"
plot "gnuplot/preMobius.de-data" u 5:4 w p, "gnuplot/preMobius.de-data" u 5:4 smooth csplines, 431000*x*x





