# set terminal postscript 12
# set output "AGS_AnaloguepEDMQy2.25-betas.eps"

set grid
set key left
set xlabel 'longitudinal coordinate'

set multiplot layout 2,1 
set ylabel "beta_x [m]" 2
p "betaFunctions" u 2:3 w l
set ylabel "beta_y [m]" 2
p "betaFunctions" u 2:4 w l
unset multiplot
