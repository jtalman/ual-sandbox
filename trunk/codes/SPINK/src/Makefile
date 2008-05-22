include $(UAL_SPINK)/src/Makefile.config

DIRS := ./SPINK


compile :
	(if [ !  -d $(UAL_SPINK)/include ]; then mkdir  $(UAL_SPINK)/include; fi;)
	(if [ !  -d $(UAL_SPINK)/lib ]; \
		then mkdir  $(UAL_SPINK)/lib; fi;)	
	(if [ !  -d $(UAL_SPINK)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_SPINK)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_SPINK)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_SPINK)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done


clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_SPINK)/include
	rm -rf $(UAL_SPINK)/lib/$(UAL_ARCH)
