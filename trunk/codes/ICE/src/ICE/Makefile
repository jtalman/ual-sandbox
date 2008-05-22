include $(UAL_SPINK)/src/Makefile.config

DIRS = ./Base ./TImpedance

OBJS = 

include ./Base/Objects
include ./TImpedance/Objects

compile : 
	(if [ !  -d $(UAL_ICE)/include/ICE ]; then mkdir  $(UAL_ICE)/include/ICE; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(DLD) $(DLDFLAGS) -o $(UAL_ICE)/lib/$(UAL_ARCH)/libIce.so $(OBJS) $(LIBS)

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_ICE)/include/ICE
	rm -f $(UAL_ICE)/lib/$(UAL_ARCH)/libIce.so
