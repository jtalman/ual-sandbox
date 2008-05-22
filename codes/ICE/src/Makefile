include $(UAL_ICE)/src/Makefile.config

DIRS := ./ICE


compile :
	(if [ !  -d $(UAL_ICE)/include ]; then mkdir  $(UAL_ICE)/include; fi;)
	(if [ !  -d $(UAL_ICE)/lib ]; \
		then mkdir  $(UAL_ICE)/lib; fi;)	
	(if [ !  -d $(UAL_ICE)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_ICE)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_ICE)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_ICE)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done


clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_ICE)/include
	rm -rf $(UAL_ICE)/lib/$(UAL_ARCH)
