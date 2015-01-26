# set terminal postscript eps enhanced color 20
# set output "timeVsBunchSize.eps"
set grid
 set xlabel 'Number of Particles in Bunch'
#set xlabel 'Bunch Length'
set ylabel 'Time (s)'
set xrange [0:40]
set yrange [0:2000]

set terminal x11 1
set title "10000 Turns E_pEDM-rtr1-preMobius.RF"
p 'gnuplot/userBunches' u 1:5 w l lw 1

set terminal x11 2
set title "10000 Turns E_Protonium"
p 'gnuplot/userBunches' u 1:6 w l lw 1

set xrange [0:20]
set yrange [0:20]

nM = 74
set terminal x11 3
set title sprintf("Number of E_pEDM-rtr1-preMobius.RF elements, nM, = %4d",nM)
p 'gnuplot/userBunches' u 1:($5/nM) w l

nP = 287
set terminal x11 4
set title sprintf("Number of E_Protonium elements, nP, = %4d",nP)
p 'gnuplot/userBunches' u 1:($6/nP) w l

set xrange [0:40]
set style line 1 lc rgb 'black' pt 5   # square
set style line 2 lc rgb 'black' pt 7   # circle
set style line 3 lc rgb 'black' pt 9   # triangle
set terminal x11 5
set title sprintf("Both Lattices, Normalized by Their Number of Elements. numMobius = %d. numPrtnm = %d",nM,nP)
p 'gnuplot/userBunches' u 1:($5/nM) w points ls 1, 'gnuplot/userBunches' u 1:($6/nP) w points ls 2, 'gnuplot/userBunches' u 1:($5/nM) w l lw 1, 'gnuplot/userBunches' u 1:($6/nP) w l lw 1

set yrange [0:100]
nMB = 16
nPB = 80
set terminal x11 6
set title sprintf("Both Lattices, Normalized by Their Number of Bends")
p 'gnuplot/userBunches' u 1:($5/nMB) w points ls 1, 'gnuplot/userBunches' u 1:($6/nPB) w points ls 2, 'gnuplot/userBunches' u 1:($5/nMB) w l lw 1, 'gnuplot/userBunches' u 1:($6/nPB) w l lw 1

nT=10000
#set key top left
#set key box
set yrange [0:0.0001]
#set label 1 "Real Time per Element ca " at 0.5,0.5
#set label "{/Symbol nu} ca 10^5, {/Symbol g}=0.2" at 0.5,0.5
#set label "frequency ca 10^5 turns/s" at 20,2.4
#set label "turn time ca 10^-5 s/turns" at 20,2.3
#set label "circumference ca 10^2 m" at 20,2.2
set terminal x11 7
set title sprintf("Two Lattices, Normalized by Number of Turns, Number of Elements and Number in Bunch. nMobius = %d. nPrtnm = %d",nM,nP)
p 'gnuplot/userBunches' u 1:( ((($5/nM)/$1))/nT ) w points ls 1, 'gnuplot/userBunches' u 1:( ((($6/nP)/$1)/nT) ) w points ls 2, 'gnuplot/userBunches' u 1:( ((($5/nM)/$1)/nT) ) w l lw 1, 'gnuplot/userBunches' u 1:( ((($6/nP)/$1)/nT) ) w l lw 1

set ylabel 'Total Computation Time (s)'
set yrange [0:5000]
set terminal png size 800,600
set output "raw_pp.png"
set title sprintf("Two Lattices, pre Mobius(16 bends), and Protonium(80 bends)\n10,000 turns")
p 'gnuplot/userBunches' u 1:5 w points ls 1, 'gnuplot/userBunches' u 1:6 w points ls 2, 'gnuplot/userBunches' u 1:5 w l lw 1, 'gnuplot/userBunches' u 1:6 w l lw 1

set ylabel 'Time/Bends/BunchNumber/Turns (s)'
set yrange [0:0.0003]
set label "computer time per bend ca 1e-4 s" at 20.5,0.00022
set label "magic velocity ca 1e+8 m/s" at 21,0.00009
set label "bend length ca 1e+1 m" at 21,0.00008
set label "real time per bend ca 1e-7 s" at 21,0.00007
#set terminal x11 8
#set term pngcairo
#set terminal png size 1024,768
#set terminal png size 3800,2400
#set terminal png size 775,600
set terminal png size 800,600
set output "pp.png"
#set terminal postscript eps enhanced
set format y '%.1e'
set title sprintf("Two Lattices, pre Mobius, and Protonium\nNormalized by Number of Bends, Number in Bunch, and Number of Turns\nnMB = %d. nPB = %d, nT = %d",nMB,nPB,nT)
p 'gnuplot/userBunches' u 1:( ((($5/nMB)/$1)/nT) ) w points ls 1, 'gnuplot/userBunches' u 1:( ((($6/nPB)/$1)/nT) ) w points ls 2, 'gnuplot/userBunches' u 1:( ((($5/nMB)/$1)/nT) ) w l lw 1, 'gnuplot/userBunches' u 1:( ((($6/nPB)/$1)/nT) ) w l lw 1
