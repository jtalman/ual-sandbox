include $(UAL)/env/$(UAL_ARCH)/Makefile.config

DIRS = ./src 

compile:  
	if [ ! -d ./$(UAL_ARCH) ] ; \
		then (mkdir ./$(UAL_ARCH); mkdir ./$(UAL_ARCH)/obj ./$(UAL_ARCH)/out; ); \
	fi;
	if [ ! -d ./lib ] ; then (mkdir ./lib; ); fi;
	if [ ! -d ./lib/$(UAL_ARCH) ] ; \
		then (mkdir ./lib/$(UAL_ARCH); mkdir ./lib/$(UAL_ARCH)/obj; ); fi;
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean: 
	@for dir in $(DIRS) ; do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done
	rm -rf ./lib/$(UAL_ARCH) ./$(UAL_ARCH)




