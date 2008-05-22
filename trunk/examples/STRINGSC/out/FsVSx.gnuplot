# % gnuplot
# gnuplot> load "FsVSx.gnuplot"

set grid

set terminal x11
# set terminal postscript
# set output "FsVSx.ps"

plot "forcecomp" using 2:6
