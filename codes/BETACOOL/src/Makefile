DIRS := ./BETACOOL


compile :
	(if [ !  -d $(UAL_BETACOOL)/include ]; then mkdir  $(UAL_BETACOOL)/include; fi;)
	(if [ !  -d $(UAL_BETACOOL)/lib ]; \
		then mkdir  $(UAL_BETACOOL)/lib; fi;)	
	(if [ !  -d $(UAL_BETACOOL)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_BETACOOL)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_BETACOOL)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_BETACOOL)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done


clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_BETACOOL)/include
	rm -rf $(UAL_BETACOOL)/lib/$(UAL_ARCH)
