# The purpose for this gnuplot file is to produce benchmark outputs
# for some of the entries in the userBunch_RT file and
# to attach 
# the p[0] p[1] p[2] p[3] p[4] p[5] entries as graph title.
# Of course you have to check that the entries in this file match
# the entries in the userBunch file.

# set terminal postscript eps enhanced color 20
# set output "SpinEvolve.eps"
set timestamp "%d/%m/%y %H:%M" font "Times, 30" rotate 4,0 
set grid

set terminal x11 1
set title "x1typ, 0, 0, 0, 0, 0"
#set xrange [0:100]
set xlabel 'turn number'
set ylabel 'p[0] = x [m]'
show timestamp
p 'out/IG' i 1 u 2:4 w l lw 1

set terminal x11 101
set title "x1typ, 0, 0, 0, 0, 0"
#set xrange [0:100]
set xlabel 'turn number'
set ylabel 'spin[0] = Sx'
show timestamp
p 'out/IG' i 1 u 2:10 w l lw 1

set terminal x11 201
set title "x1typ, 0, 0, 0, 0, 0"
#set xrange [0:100]
set xlabel 'turn number'
set ylabel 'spin[1] = Sy'
show timestamp
p 'out/IG' i 1 u 2:11 w l lw 1

set terminal x11 3
set title "0, x2typ, 0, 0, 0, 0"
#set xrange [0:100]
set xlabel 'turn number'
set ylabel "p[0] = x [m]"
show timestamp
p 'out/IG' i 3 u 2:5 w l lw 1

set terminal x11 5
set title "0, 0, y1typ, 0, 0, 0"
#set xrange [0:100]
set xlabel 'turn number'
set ylabel 'p[2] = y [m]'
show timestamp
p 'out/IG' i 5 u 2:7 w l lw 1

set terminal x11 7
set title "0, 0, 0, y2typ, 0, 0"
#set xrange [0:100]
set xlabel 'turn number'
set ylabel "p[2] = y [m]"
show timestamp
p 'out/IG' i 7 u 2:8 w l lw 1

set terminal x11 9
set title "0, 0, 0, 0, 0, deltyp"
#set xrange [0:100]
set xlabel 'turn number'
set ylabel '-ct [m]'
show timestamp
p 'out/IG' i 9 u 2:9 w l lw 1

set terminal x11 101112
set title "0, 0, 0, 0, 0, -0.000001/-0.000002"
#set xrange [*:*]
set xlabel 'x'
set ylabel '-ct [m]'
show timestamp
p 'out/IG' i 10 u 8:9 w l lw 1, 'out/IG' i 11 u 8:9 w l lw 1, 'out/IG' i 12 u 8:9 w l lw 1

