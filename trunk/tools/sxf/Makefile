include ./config/$(SXF_ARCH)/Makefile.config

DIRS = ./src ./examples ./api # ./doc

compile:  
	@for dir in $(DIRS); do \
	(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean: 
	@for dir in $(DIRS) ; do \
	(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done

