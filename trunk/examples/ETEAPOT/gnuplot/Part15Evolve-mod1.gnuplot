# The purpose for this gnuplot file is to produce numerous plots for each
# Particle 15 in the userBunch file and to attach line 15 as title 

set terminal png large size 800,600
set grid

set terminal x11 2
# set output "eps/15/Part15.i:x.png"
set title "index:x   0, 0, 0.005, 0, 0, 0"
set xlabel 'turn index'
set ylabel 'x [m]'
p 'IG' i 15 u 2:4 w l lw 1

set terminal x11 3
# set output "eps/15/Part15.i:y.png"
set title "index:y   0, 0, 0.005, 0, 0, 0"
set xlabel 'turn index'
set ylabel 'y [m]'
p 'IG' i 15 u 2:6 w l lw 1

set terminal x11 4
# set output "eps/15/Part15.ct:de.png"
set title "ct:de   0, 0, 0.005, 0, 0, 0"
set xlabel 'ct'
set ylabel 'de'
p 'IG' i 15 u 8:9 w l lw 1

set terminal x11 1
# set output "eps/15/Part15.i:Sx.png"
set title "index:Sx   0, 0, 0.005, 0, 0, 0"
set xlabel 'turn index'
set ylabel 'S[0]'
f(x) = a + b*x + c*x*x
fit f(x) 'IG' i 15 u 2:10:(1.0e-6) via a,b,c
w = -pi*c/b
A = a - b*b/2/c + b*b/4/c
x0 = asin(a/A)

g(x) = y0 + A*sin(w*x-x0) 
y0 = a
FIT_LIMIT = 0.002
fit g(x) 'IG' i 15 u 2:10:(1.0e-6) via w,A,x0,y0

set label 1 sprintf("x0 = %3.4g",x0) at graph -0.10,-0.08
set label 2 sprintf("w = %3.4g",w) at graph  0.08,-0.08
set label 3 sprintf("A = %3.4g",A) at graph  0.26,-0.08
set label 4 sprintf("y0 = %3.4g",y0) at graph  0.60,-0.08

show label 1
show label 2
show label 3
show label 4

p 'IG' i 15 u 2:10 w l lw 1, g(x)
