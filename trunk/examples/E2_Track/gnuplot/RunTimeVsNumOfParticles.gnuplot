#set terminal png giant size 800,600
#set output "RunTimeVsNumOfParticles.png"
set timestamp "%d/%m/%y %H:%M" rotate 4,0
set xlabel 'number of particles in bunch'
set ylabel 'run time [s]'
set grid
f(x) = a + b*x
fit f(x) "gnuplot/userBunches" u 1:5 via a,b
p "gnuplot/userBunches" u 1:5 w p, 2.57214 + 25.8344*x
