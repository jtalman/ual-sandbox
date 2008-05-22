# % gnuplot
# gnuplot> load "../twiss.gnuplot"

set xrange [0:15]
set yrange [0:25]

# set terminal x11
set terminal postscript
set output "twiss.ps"

plot "twiss" using 3:4 with linespoints, "twiss" using 3:8 with linespoints
#     "twiss" using 3:(-100*$7) with linespoints
