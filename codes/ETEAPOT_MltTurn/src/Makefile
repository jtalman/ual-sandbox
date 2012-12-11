include $(UAL_ETEAPOT_MltTurn)/src/Makefile.config

DIRS =  ./ETEAPOT_MltTurn

LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacSMF
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacOptics 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacSurvey
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) -lZTps
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH)		-lgsl -lgslcblas

include ./ETEAPOT_MltTurn/Integrator/Objects

compile : 
	if [ ! -d $(UAL_ETEAPOT_MltTurn)/include ] ; \
		then (mkdir $(UAL_ETEAPOT_MltTurn)/include;); fi;
	(if [ !  -d $(UAL_ETEAPOT_MltTurn)/lib ]; \
		then mkdir  $(UAL_ETEAPOT_MltTurn)/lib; fi;)
	(if [ !  -d $(UAL_ETEAPOT_MltTurn)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_ETEAPOT_MltTurn)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_ETEAPOT_MltTurn)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_ETEAPOT_MltTurn)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(LD) $(LDFLAGS) $(INC) -o $(UAL_ETEAPOT_MltTurn)/lib/$(UAL_ARCH)/obj/dummy   \
		./dummy.cc $(OBJS) $(LIBS)
	cp /dev/null $(UAL_ETEAPOT_MltTurn)/lib/$(UAL_ARCH)/obj/dummy
	$(DLD) $(DLDFLAGS) -o $(UAL_ETEAPOT_MltTurn)/lib/$(UAL_ARCH)/libETEAPOT_MltTurn.so $(OBJS) $(LIBS)

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_ETEAPOT_MltTurn)/include
	rm -rf $(UAL_ETEAPOT_MltTurn)/lib/$(UAL_ARCH)
