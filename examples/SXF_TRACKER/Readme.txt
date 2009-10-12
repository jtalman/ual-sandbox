Build:
   make clean
   make

   --->>> Objects in ./lib/linux/obj/dipole.o
                                     driftOrDefault.o
                                     positionPrinter.o
                                     rfCavity.o
                                     timer.o
                                     tracker.o

   --->>> Executable in ./tracker

Run:
   ./tracker data/sxf_tracker.apdf

   --->>> Ouput in ./out/cpp/...map1
                             ...orbit
                             ...sxf
                             ...twiss
           
                   stdout
                   stderr

Notes:
   The built executable
      ./tracker
   implicitly uses data file
      ./data/muon0.13_R5m.sxf
