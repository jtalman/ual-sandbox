include $(UAL)/env/$(UAL_ARCH)/Makefile.config

DIRS = ./src 

compile:  	
	if [ ! -d ./bin ] ; then mkdir ./bin ; fi;
	if [ ! -d ./bin/$(UAL_ARCH) ] ; then mkdir ./bin/$(UAL_ARCH) ; fi;
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean: 
	@for dir in $(DIRS) ; do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done
	rm -rf  ./bin/$(UAL_ARCH)

install:  
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) install; fi;); \
	done