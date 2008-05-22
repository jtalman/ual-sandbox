include $(UAL_SIMBAD)/src/Makefile.config

DIRS = ./Base ./Tracker ./SC

# ifdef MPIHOME
# DIRS += ./SC_MPI
# DIRS += ./SC3D_MPI
# LIBS += -L$(MPIHOME)/lib -lmpich
# endif

OBJS = 

include ./Base/Objects
include ./Tracker/Objects
include ./SC/Objects

# ifdef MPIHOME
# include ./SC_MPI/Objects
# include ./SC3D_MPI/Objects
# endif

compile : 
	(if [ !  -d $(UAL_SIMBAD)/include/SIMBAD ]; then mkdir  $(UAL_SIMBAD)/include/SIMBAD; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(DLD) $(DLDFLAGS) -o $(UAL_SIMBAD)/lib/$(UAL_ARCH)/libSimbad.so $(OBJS) $(LIBS)

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_SIMBAD)/include/SIMBAD
	rm -f $(UAL_SIMBAD)/lib/$(UAL_ARCH)/libSimbad.so
