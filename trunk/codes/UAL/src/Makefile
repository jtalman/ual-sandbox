include $(UAL)/env/$(UAL_ARCH)/Makefile.config

DIRS = ./UAL

compile :
	(if [ !  -d $(UAL_CORE)/include ]; then mkdir  $(UAL_CORE)/include; fi;)
	(if [ !  -d $(UAL_CORE)/lib ]; \
		then mkdir  $(UAL_CORE)/lib; fi;)	
	(if [ !  -d $(UAL_CORE)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_CORE)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_CORE)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_CORE)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done


clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_CORE)/include
	rm -rf $(UAL_CORE)/lib/$(UAL_ARCH)