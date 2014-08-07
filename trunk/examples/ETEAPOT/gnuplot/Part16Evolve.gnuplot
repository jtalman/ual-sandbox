# The purpose for this gnuplot file is to produce numerous plots for each
# Particle 16 in the userBunch file and to attach line 16 as title 

set grid

set terminal x11 2
# set output "Part16.i:x.eps"
set title "index:x   0, 0, 0.01, 0, 0, 0"
set xlabel 'turn index'
set ylabel 'x [m]'
p 'IG' i 16 u 2:4 w l lw 1

set terminal x11 3
# set output "Part16.i:y.eps"
set title "index:y   0, 0, 0.01, 0, 0, 0"
set xlabel 'turn index'
set ylabel 'y [m]'
p 'IG' i 16 u 2:6 w l lw 1

set terminal x11 4
# set output "Part16.ct:de.eps"
set title "ct:de   0, 0, 0.01, 0, 0, 0"
set xlabel 'ct'
set ylabel 'de'
p 'IG' i 16 u 8:9 w l lw 1

set terminal x11 1
# set output "Part16.i:Sx.eps"
set title "index:Sx   0, 0, 0.01, 0, 0, 0"
set xlabel 'turn index'
set ylabel 'S[0]'
f(x) = a + b*x + c*x*x
fit f(x) 'IG' i 16 u 2:10:(1.0e-6) via a,b,c
set label 1 sprintf("a = %3.4g",a) at graph -0.15,-0.08
set label 2 sprintf("b = %3.4g",b) at graph 0.03,-0.08
set label 3 sprintf("c = %3.4g",c) at graph 0.21,-0.08
show label 1
show label 2
show label 3
p 'IG' i 16 u 2:10 w l lw 1, f(x)
