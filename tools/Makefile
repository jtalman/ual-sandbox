
DIRS = ./tars ./sxf

# ifdef MPIHOME
# DIRS += ./shortmpi
# endif

# ifdef ROOTSYS
# CORAANT=$(UAL)/tools/coraant
# endif

compile:
	if [ ! -d ./lib ] ; then (mkdir ./lib; ); fi;
	if [ ! -d ./lib/$(UAL_ARCH) ] ; then (mkdir ./lib/$(UAL_ARCH); ); fi;
	if [ ! -d ./bin ] ; then (mkdir ./bin; ); fi;
	if [ ! -d ./bin/$(UAL_ARCH) ] ; then (mkdir ./bin/$(UAL_ARCH); ); fi;
	if [ ! -d ./include ] ; then (mkdir ./include; ); fi;
	if [ ! -d ./man ] ; then (mkdir ./man; ); fi;
	if [ ! -d ./info ] ; then (mkdir ./info; ); fi;
	if [ ! -d ./doc ] ; then (mkdir ./doc; ); fi;
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
#
#	if [ $(CORAANT) ] ; then \
#		(cd $(CORAANT); \
#		configure --with-fftwinc=$(UAL)/tools/include --with-fftwlib=$(UAL)/tools/lib/linux --includedir=$(UAL)/tools/include --libdir=$(UAL)/tools/lib/linux; \
#		make;  \
#		cd ..); \
#	fi;


clean:
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done
	(rm -rf gperf/$(UAL_ARCH))
	rm -rf ./lib/$(UAL_ARCH) ./bin/$(UAL_ARCH) ./man ./include ./info ./doc



