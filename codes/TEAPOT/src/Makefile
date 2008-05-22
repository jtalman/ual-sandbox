include $(UAL_TEAPOT)/src/Makefile.config

DIRS =  ./Math  ./Integrator  ./Main ./TEAPOT

LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacSMF
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacOptics 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacSurvey
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) -lZTps
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH)		-lgsl -lgslcblas

include ./Math/Objects
include ./Integrator/Objects
include ./Main/Objects
include ./TEAPOT/Integrator/Objects
include ./TEAPOT/StringSC/Objects

compile : 
	if [ ! -d $(UAL_TEAPOT)/include ] ; \
		then (mkdir $(UAL_TEAPOT)/include;); fi;
	(if [ !  -d $(UAL_TEAPOT)/lib ]; \
		then mkdir  $(UAL_TEAPOT)/lib; fi;)	
	(if [ !  -d $(UAL_TEAPOT)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_TEAPOT)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_TEAPOT)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_TEAPOT)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(LD) $(LDFLAGS) $(INC) -o $(UAL_TEAPOT)/lib/$(UAL_ARCH)/obj/dummy   \
		./dummy.cc $(OBJS) $(LIBS)
	cp /dev/null $(UAL_TEAPOT)/lib/$(UAL_ARCH)/obj/dummy
	$(DLD) $(DLDFLAGS) -o $(UAL_TEAPOT)/lib/$(UAL_ARCH)/libTeapot.so $(OBJS) $(LIBS)

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_TEAPOT)/include
	rm -rf $(UAL_TEAPOT)/lib/$(UAL_ARCH)



