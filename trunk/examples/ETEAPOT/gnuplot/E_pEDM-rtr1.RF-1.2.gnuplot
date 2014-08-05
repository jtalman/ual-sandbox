# gnuplot> load "gnuplot/pEDM-rtr1.RF-1.2.gnuplot"

set grid
set pointsize 3

set xlabel "x_{max} [m]"
set ylabel "s_x"
set title 'E\_pEDM.rtr1-x-sx, m=-1.2, 63470 turns'
set terminal postscript eps enhanced 20 
set output "eps/E\_pEDM.rtr1-x-sx.eps"
plot "gnuplot/E_pEDM-rtr1.RF-1.2.x-data" u 4:5 w p, "gnuplot/E_pEDM-rtr1.RF-1.2.x-data" u 4:5 smooth csplines, -240*x*x

set xlabel "y_{max} [m]"
set ylabel "s_x"
set title 'E\_pEDM.rtr1-y-sx, m=-1.2, 63470 turns'
set terminal postscript eps enhanced 20 
set output "eps/E\_pEDM.rtr1-y-sx.eps"
plot "gnuplot/E_pEDM-rtr1.RF-1.2.y-data" u 4:5 w p, "gnuplot/E_pEDM-rtr1.RF-1.2.y-data" u 4:5 smooth csplines, 300*x*x

set xlabel "de_0"
set ylabel "s_x"
set title 'E\_pEDM.rtr1-de-sx, m=-1.2, 63470 turns'
set terminal postscript eps enhanced 20 
set output "eps/E\_pEDM.rtr1-de-sx.eps"
plot "gnuplot/E_pEDM-rtr1.RF-1.2.de-data" u 2:5 w p, "gnuplot/E_pEDM-rtr1.RF-1.2.de-data" u 2:5 smooth csplines, 650000*x*x





