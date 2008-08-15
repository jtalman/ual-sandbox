include $(UAL)/env/$(UAL_ARCH)/Makefile.config

DIRS = ./src ./api # ./doc #./examples

compile:  
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean: 
	@for dir in $(DIRS) ; do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done


