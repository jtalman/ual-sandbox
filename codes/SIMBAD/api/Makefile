include $(UAL)/env/$(UAL_ARCH)/Makefile.config

ifdef ROOTSYS
DIRS += ./ROOT

compile :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile.PL ]; then $(PERL) Makefile.PL; $(MAKE);\
			elif [ -f ./Makefile ]; then $(MAKE); fi;);\
	done
clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf blib;

else
compile:

clean:

endif
