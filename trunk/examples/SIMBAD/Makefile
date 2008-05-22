
DIRS = ./src 

compile:  
	if [ ! -d ./lib/$(UAL_ARCH) ] ; \
		then (mkdir ./lib/; mkdir ./lib/$(UAL_ARCH); mkdir ./lib/$(UAL_ARCH)/obj; ); fi;
	if [ ! -d ./$(UAL_ARCH) ] ; \
		then (mkdir ./$(UAL_ARCH); ); fi;
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean: 
	@for dir in $(DIRS) ; do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done
	rm -rf ./lib/$(UAL_ARCH) ./$(UAL_ARCH)
