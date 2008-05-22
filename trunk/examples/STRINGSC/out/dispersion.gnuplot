# % gnuplot
# gnuplot> load "../dispersion.gnuplot"

set xrange [0:15]
# set yrange [0:40]

# set terminal x11
set terminal postscript
set output "dispersion.ps"

plot "twiss" using 3:7 with linespoints
