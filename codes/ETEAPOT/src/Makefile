include $(UAL_ETEAPOT)/src/Makefile.config

DIRS =  ./ETEAPOT ./Main

LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacSMF
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacOptics 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacSurvey
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) -lZTps
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH)		-lgsl -lgslcblas

include ./ETEAPOT/Integrator/Objects ./Main/Objects

compile : 
	if [ ! -d $(UAL_ETEAPOT)/include ] ; \
		then (mkdir $(UAL_ETEAPOT)/include;); fi;
	(if [ !  -d $(UAL_ETEAPOT)/lib ]; \
		then mkdir  $(UAL_ETEAPOT)/lib; fi;)
	(if [ !  -d $(UAL_ETEAPOT)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_ETEAPOT)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_ETEAPOT)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_ETEAPOT)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(LD) $(LDFLAGS) $(INC) -o $(UAL_ETEAPOT)/lib/$(UAL_ARCH)/obj/dummy   \
		./dummy.cc $(OBJS) $(LIBS)
	cp /dev/null $(UAL_ETEAPOT)/lib/$(UAL_ARCH)/obj/dummy
	$(DLD) $(DLDFLAGS) -o $(UAL_ETEAPOT)/lib/$(UAL_ARCH)/libETeapot.so $(OBJS) $(LIBS)

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_ETEAPOT)/include
	rm -rf $(UAL_ETEAPOT)/lib/$(UAL_ARCH)
