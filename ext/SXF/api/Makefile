include $(UAL)/env/$(UAL_ARCH)/Makefile.config

DIRS = ./UAL/SXF/XS

compile :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile.PL ]; then $(PERL) Makefile.PL; $(MAKE); fi;); \
	done
clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 	
	rm -rf ./blib/$(UAL_ARCH);

