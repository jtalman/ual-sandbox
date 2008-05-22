include ./Makefile.config

DIRS := ./AIM


compile :
	(if [ !  -d $(UAL_AIM)/include ]; \
		 then mkdir  $(UAL_AIM)/include; fi;)
	(if [ !  -d $(UAL_AIM)/lib ]; \
		then mkdir  $(UAL_AIM)/lib; fi;)	
	(if [ !  -d $(UAL_AIM)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_AIM)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_AIM)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_AIM)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done


clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_AIM)/include
	rm -rf $(UAL_AIM)/lib/$(UAL_ARCH)
