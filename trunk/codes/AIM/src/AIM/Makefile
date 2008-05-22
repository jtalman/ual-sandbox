include $(UAL_AIM)/src/Makefile.config

DIRS = ./Common ./BTF ./BPM

OBJS = 
include ./Common/Objects
include ./BTF/Objects 
include ./BPM/Objects 

compile : 
	if [ ! -d $(UAL_AIM)/include/AIM ] ; \
		then mkdir $(UAL_AIM)/include/AIM ; fi;
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(DLD) $(DLDFLAGS) -o $(UAL_AIM)/lib/$(UAL_ARCH)/libAim.so $(OBJS) $(LIBS)

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -f $(UAL_AIM)/lib/$(UAL_ARCH)/libAim.so
	rm -rf $(UAL_AIM)/include/AIM

