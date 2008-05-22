include $(UAL)/ext/UI/src/Makefile.config

DIRS = ./UI

OBJS = 
include ./UI/Objects

compile:
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(DLD) $(DLDFLAGS) -o $(UAL)/ext/UI/lib/$(UAL_ARCH)/libUalUI.so $(OBJS) $(LIBS)

clean:
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 	
	rm -f $(UAL)/ext/UI/lib/$(UAL_ARCH)/libUalUI.so


