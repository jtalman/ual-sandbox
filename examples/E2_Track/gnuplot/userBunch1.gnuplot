# set terminal postscript eps enhanced color 20
# set output "SpinEvolve.eps"
set grid
set xlabel 'split bend index'

#set format x "%s*10^{%S}"
#set format x "%E"
 set format x "%4.2E"
set rmargin 8.5

set ylabel 'p[0] = x'
set terminal x11 0
set title "+x1typ, 0, 0, 0, 0, 0"
p 'out/IG' i 1 u 2:4 w l lw 1

set ylabel 'p[1] = dx/dz'
set terminal x11 1
set title "+x1typ, 0, 0, 0, 0, 0"
p 'out/IG' i 1 u 2:5 w l lw 1

set ylabel 'p[2] = y'
set terminal x11 2
set title "+x1typ, 0, 0, 0, 0, 0"
p 'out/IG' i 1 u 2:6 w l lw 1

set ylabel 'p[3] = dy/dz'
set terminal x11 3
set title "+x1typ, 0, 0, 0, 0, 0"
p 'out/IG' i 1 u 2:7 w l lw 1

set ylabel 'p[4] = -c delta t'
set terminal x11 4
set title "+x1typ, 0, 0, 0, 0, 0"
p 'out/IG' i 1 u 2:8 w l lw 1

set ylabel 'p[5] = delta E / pDc'
set terminal x11 5
set title "+x1typ, 0, 0, 0, 0, 0"
p 'out/IG' i 1 u 2:9 w l lw 1

set ylabel 'S[0] = Sx'
set terminal x11 6 size 800,600 position 400,200 
set title "+x1typ, 0, 0, 0, 0, 0"
p 'out/IG' i 1 u 2:10 w l lw 1

set format y "%12.10E"
set ylabel 'Lxc'
set terminal x11 7
set title "+x1typ, 0, 0, 0, 0, 0"
p 'out/IG' i 1 u 2:13 w l lw 1

set ylabel 'Lyc'
set terminal x11 8
set title "+x1typ, 0, 0, 0, 0, 0"
p 'out/IG' i 1 u 2:14 w l lw 1

set ylabel 'Lzc'
set terminal x11 9
set title "+x1typ, 0, 0, 0, 0, 0"
p 'out/IG' i 1 u 2:15 w l lw 1

set ylabel 'Lc'
set terminal x11 10
set title "+x1typ, 0, 0, 0, 0, 0"
p 'out/IG' i 1 u 2:16 w l lw 1

# vi $UAL/codes/ETEAPOT2/src/ETEAPOT2/Integrator/bendMethods/traverseSplitBendExactly +80
