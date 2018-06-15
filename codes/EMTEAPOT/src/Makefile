include $(UAL_EMTEAPOT)/src/Makefile.config

DIRS =  ./EMTEAPOT

LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacSMF
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacOptics 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacSurvey
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) -lZTps
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH)		-lgsl -lgslcblas

include ./EMTEAPOT/Integrator/Objects

compile : 
	if [ ! -d $(UAL_EMTEAPOT)/include ] ; \
		then (mkdir $(UAL_EMTEAPOT)/include;); fi;
	(if [ !  -d $(UAL_EMTEAPOT)/lib ]; \
		then mkdir  $(UAL_EMTEAPOT)/lib; fi;)
	(if [ !  -d $(UAL_EMTEAPOT)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_EMTEAPOT)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_EMTEAPOT)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_EMTEAPOT)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(LD) $(LDFLAGS) $(INC) -o $(UAL_EMTEAPOT)/lib/$(UAL_ARCH)/obj/dummy   \
		./dummy.cc $(OBJS) $(LIBS)
	cp /dev/null $(UAL_EMTEAPOT)/lib/$(UAL_ARCH)/obj/dummy
	$(DLD) $(DLDFLAGS) -o $(UAL_EMTEAPOT)/lib/$(UAL_ARCH)/libEMTEAPOT.so $(OBJS) $(LIBS)

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_EMTEAPOT)/include
	rm -rf $(UAL_EMTEAPOT)/lib/$(UAL_ARCH)
