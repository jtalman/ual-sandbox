# % gnuplot
# gnuplot> load "FsVSxANDct.gnuplot"

set view 70, 10

set terminal x11
# set terminal postscript
# set output "FsVSxANDct.ps"

splot "forcecomp" using 3:2:6
