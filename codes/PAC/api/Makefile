include $(UAL)/env/$(UAL_ARCH)/Makefile.config

DIRS = ./Beam ./Survey ./Optics ./SMF

# ifdef ROOTSYS
# DIRS += ./ROOT
# endif

compile:
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile.PL ]; then $(PERL) Makefile.PL; $(MAKE); \
		elif [ -f ./Makefile ]; then $(MAKE);  fi;); \
	done
clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 	
	rm -rf ./blib/$(UAL_ARCH);


