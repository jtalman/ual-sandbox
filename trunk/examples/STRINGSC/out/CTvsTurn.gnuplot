# % gnuplot
# gnuplot> load "CTvsTurn.gnuplot"

set grid

set terminal x11
# set terminal postscript
# set output "CTvsTurn.ps"

plot "forcecomp" using 2
