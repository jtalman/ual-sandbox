include $(UAL_ETEAPOT2)/src/Makefile.config

DIRS =  ./ETEAPOT2

LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacSMF
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacOptics 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacSurvey
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) -lZTps
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH)		-lgsl -lgslcblas

include ./ETEAPOT2/Integrator/Objects

compile : 
	if [ ! -d $(UAL_ETEAPOT2)/include ] ; \
		then (mkdir $(UAL_ETEAPOT2)/include;); fi;
	(if [ !  -d $(UAL_ETEAPOT2)/lib ]; \
		then mkdir  $(UAL_ETEAPOT2)/lib; fi;)
	(if [ !  -d $(UAL_ETEAPOT2)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_ETEAPOT2)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_ETEAPOT2)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_ETEAPOT2)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(LD) $(LDFLAGS) $(INC) -o $(UAL_ETEAPOT2)/lib/$(UAL_ARCH)/obj/dummy   \
		./dummy.cc $(OBJS) $(LIBS)
	cp /dev/null $(UAL_ETEAPOT2)/lib/$(UAL_ARCH)/obj/dummy
	$(DLD) $(DLDFLAGS) -o $(UAL_ETEAPOT2)/lib/$(UAL_ARCH)/libETeapot2.so $(OBJS) $(LIBS)

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_ETEAPOT2)/include
	rm -rf $(UAL_ETEAPOT2)/lib/$(UAL_ARCH)
