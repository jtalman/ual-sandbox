include $(UAL)/ext/ADXF/src/Makefile.config

DIRS = ./UAL 

compile :
	(if [ !  -d $(UAL)/ext/ADXF/include ]; then mkdir  $(UAL)/ext/ADXF/include; fi;)
	(if [ !  -d $(UAL)/ext/ADXF/lib ]; then mkdir  $(UAL)/ext/ADXF/lib; fi;)
	if [ ! -d ../lib/$(UAL_ARCH) ] ; \
		then (mkdir  ../lib/$(UAL_ARCH); \
		mkdir  ../lib/$(UAL_ARCH)/obj;   \
		mkdir  ../lib/$(UAL_ARCH)/obj/elements; \
		mkdir  ../lib/$(UAL_ARCH)/obj/handlers; ); \
	fi;	
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf ../lib/$(UAL_ARCH)

