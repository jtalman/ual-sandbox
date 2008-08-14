include $(UAL)/env/$(UAL_ARCH)/Makefile.config

DIRS = ./src ./api # ./doc 

compile:  
	if [ ! -d ./lib ] ; then (mkdir ./lib; ); fi;
	if [ ! -d ./lib/$(UAL_ARCH) ] ; \
		then (mkdir ./lib/$(UAL_ARCH); mkdir ./lib/$(UAL_ARCH)/obj; ); fi;
#	if [ "$(origin UAL_JAVA)" == "environment" ];  then ant -buildfile build.xml jar; fi;
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean: 
	@for dir in $(DIRS) ; do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done
	rm -rf ./lib/$(UAL_ARCH)
#	if [ "$(origin UAL_JAVA)" == "environment" ];  then ant -buildfile build.xml clean; fi;



