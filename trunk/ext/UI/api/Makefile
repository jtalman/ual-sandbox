include $(UAL)/env/$(UAL_ARCH)/Makefile.config

ifdef ROOTSYS
	DIRS = ./ROOT

compile :

	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 

else
compile:

clean:

endif

