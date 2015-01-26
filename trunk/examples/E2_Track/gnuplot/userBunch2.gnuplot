# set terminal postscript eps enhanced color 20
# set output "SpinEvolve.eps"
set grid
set xlabel 'turn index'
set ylabel 'S[0]'

set terminal x11 1
set title "+x1typ, 0, 0, 0, 0, 0"
p 'out/IG' i 1 u 2:10 w l lw 1

set terminal x11 2
set title "-x1typ,  0, 0, 0, 0, 0"
p 'out/IG' i 2 u 2:10 w l lw 1
