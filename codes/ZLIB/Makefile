include $(UAL)/env/$(UAL_ARCH)/Makefile.config

DIRS = ./src ./api ./examples # ./doc


compile:  
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean: 
	rm -f ../zlib_$(UAL_ARCH).tar
	@for dir in $(DIRS) ; do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done

