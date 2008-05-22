include $(UAL)/env/$(UAL_ARCH)/Makefile.config

DIRS := ./TIBETAN


compile :
	(if [ !  -d $(UAL_TIBETAN)/include ]; \
		 then mkdir  $(UAL_TIBETAN)/include; fi;)
	(if [ !  -d $(UAL_TIBETAN)/lib ]; \
		then mkdir  $(UAL_TIBETAN)/lib; fi;)	
	(if [ !  -d $(UAL_TIBETAN)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_TIBETAN)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_TIBETAN)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_TIBETAN)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done


clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_TIBETAN)/include
	rm -rf $(UAL_TIBETAN)/lib/$(UAL_ARCH)
