include $(UAL_SXF)/src/Makefile.config

DIRS = ./UAL # ./ual_sxf

compile :
	(if [ !  -d $(UAL_SXF)/include ]; then mkdir  $(UAL_SXF)/include; fi;)
	if [ ! -d ../lib/$(UAL_ARCH) ] ; \
		then (mkdir  ../lib/$(UAL_ARCH); \
		mkdir  ../lib/$(UAL_ARCH)/obj; \
		mkdir  ../lib/$(UAL_ARCH)/obj/buckets; ); \
	fi;	
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf ../lib/$(UAL_ARCH)

