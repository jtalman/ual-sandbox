set grid
set xlabel 'turn index'
set ylabel 'x [m]'

set terminal x11 0
set terminal postscript eps dl 3 enhanced color 20
set output "eps/xEvolve0.P1.0.eps"
set title "P1.0-0: 0.01, 0, 0, 0, 0, 5.85e-5"
p 'savIG-P1.0-0' i 0 u 2:4 w l lw 2

set output "eps/xEvolve0.P1.0-200turn.eps"
set xrange [0:200]
set title "P1.0-0: 0.01, 0, 0, 0, 0, 5.85e-5"
p 'savIG-P1.0-0' i 0 u 2:4 w l lw 2
