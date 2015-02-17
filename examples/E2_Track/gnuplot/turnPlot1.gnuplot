# set terminal postscript eps enhanced color 20
# set output "SpinEvolve.eps"
set grid
set xlabel 'turn index'

#set format x "%s*10^{%S}"
#set format x "%E"
 set format x "%4.2E"
set rmargin 8.5

set ylabel 'p[0] = x'
set terminal x11 10
set title "+x1typ, 0, 0, 0, 0, 0"
p 'NikolayOut' u 2:4 w l lw 1

set ylabel 'p[1] = dx/dz'
set terminal x11 11
set title "+x1typ, 0, 0, 0, 0, 0"
p 'NikolayOut' u 2:5 w l lw 1

set ylabel 'p[2] = y'
set terminal x11 12
set title "+x1typ, 0, 0, 0, 0, 0"
p 'NikolayOut' u 2:6 w l lw 1

set ylabel 'p[3] = dy/dz'
set terminal x11 13
set title "+x1typ, 0, 0, 0, 0, 0"
p 'NikolayOut' u 2:7 w l lw 1

set ylabel 'p[4] = -c delta t'
set terminal x11 14
set title "+x1typ, 0, 0, 0, 0, 0"
p 'NikolayOut' u 2:8 w l lw 1

set ylabel 'p[5] = delta E / pDc'
set terminal x11 15
set title "+x1typ, 0, 0, 0, 0, 0"
p 'NikolayOut' u 2:9 w l lw 1

set ylabel 'S[0] = Sx'
set terminal x11 16 size 800,600 position 400,200 
set title "+x1typ, 0, 0, 0, 0, 0"
p 'NikolayOut' u 2:10 w l lw 1
