# The purpose for this gnuplot file is to produce one plot for each
# line in the userBunch file and, for lines 10 through 18 to attach 
# the p[0] p[1] p[2] p[3] p[4] p[5] entries as graph title.
# Of course you have to check that the entries in this file match
# the entries in the userBunch file.

set grid
set xlabel 'turn index'
set ylabel 's[0]'

set terminal x11 11
set terminal postscript eps dl 3 enhanced color 20
set output "eps/SpinEvolve11.P1.0.eps"
set title "P1.0: 0.005, 0, 0, 0, 0, 0"
p 'IG' i 11 u 2:10 w l lw 1, -0.00157/10000*x lt 3 lw 3

set terminal x11 12
set terminal postscript eps dl 3 enhanced color 20
set output "eps/SpinEvolve12.P1.0.eps"
set title "P1.0: -0.005, 0, 0, 0, 0, 0"
p 'IG' i 12 u 2:10 w l lw 1, -0.00157/10000*x lt 3 lw 3

set terminal x11 13
set terminal postscript eps dl 3 enhanced color 20
set output "eps/SpinEvolve13.P1.0.eps"
set title "P1.0: 0, 0.0002, 0, 0, 0, 0"
p 'IG' i 13 u 2:10 w l lw 1, -0.00062*x/1000 lt 3 lw 3

set terminal x11 14
set terminal postscript eps dl 3 enhanced color 20
set output "eps/SpinEvolve14.P1.0.eps"
set title "P1.0: 0, -0.0002, 0, 0, 0, 0"
p 'IG' i 14 u 2:10 w l lw 1, -0.0062/10000*x lt 3 lw 3

set terminal x11 15
set terminal postscript eps dl 3 enhanced color 20
set output "eps/SpinEvolve15.P1.0.eps"
set title "P1.0: 0, 0, 0.01, 0, 0, 0"
p 'IG' i 15 u 2:10 w l lw 1, 0.00245/10000*x lt 3 lw 3

set terminal x11 16
set terminal postscript eps dl 3 enhanced color 20
set output "eps/SpinEvolve16.P1.0.eps"
set title "P1.0: 0, 0, 0, 0.001, 0, 0"
p 'IG' i 16 u 2:10 w l lw 1, 0.037/10000*x lt 3 lw 3

set terminal x11 17
set terminal postscript eps dl 3 enhanced color 20
set output "eps/SpinEvolve17.P1.0.eps"
set yrange [-1:1]
set title "P1.0: 0, 0, 0, 0, 0, 5.85e-5"
p 'IG' i 17 u 2:10 w l lw 1, x/600 lt 3 lw 3

set terminal x11 18
set terminal postscript eps dl 3 enhanced color 20
set output "eps/SpinEvolve18.P1.0.eps"
set title "P1.0: 0, 0, 0, 0, 0,-5.85e-5"
p 'IG' i 18 u 2:10 w l lw 1, -x/600 lt 3 lw 3

set terminal x11 0
set terminal postscript eps dl 3 enhanced color 20
set output "eps/SpinEvolve0.P1.0.eps"
set title "P1.0: 0.01, 0, 0, 0, 0, 5.85e-5"
p 'IG' i 0 u 2:10 w l lw 1, x/600 lt 3 lw 3

