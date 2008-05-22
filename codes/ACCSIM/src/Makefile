include $(UAL_ACCSIM)/src/Makefile.config

DIRS = ./ACCSIM

compile :
	(if [ !  -d $(UAL_ACCSIM)/include ]; \
		then mkdir  $(UAL_ACCSIM)/include; fi;)
	(if [ !  -d $(UAL_ACCSIM)/include/ACCSIM ]; \
		then mkdir  $(UAL_ACCSIM)/include/ACCSIM; fi;)
	(if [ !  -d $(UAL_ACCSIM)/lib ]; \
		then mkdir  $(UAL_ACCSIM)/lib; fi;)	
	(if [ !  -d $(UAL_ACCSIM)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_ACCSIM)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_ACCSIM)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_ACCSIM)/lib/$(UAL_ARCH)/obj; fi;)	
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_ACCSIM)/include/ACCSIM
	rm -rf $(UAL_ACCSIM)/lib/$(UAL_ARCH)
