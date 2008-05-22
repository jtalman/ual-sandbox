DIRS = ./QT # ./ROOT # ./COIN3D

ifdef ROOTSYS
	DIRS += ./ROOT
endif


compile:  
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean: 
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done
