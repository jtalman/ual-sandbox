# % gnuplot
# gnuplot> load "FsVSct.gnuplot"

set grid

set terminal x11
# set terminal postscript
# set output "FsVSct.ps"

plot "forcecomp" using 3:6
