include $(UAL)/ext/UI/src/Makefile.config

DIRS = ./UAL

compile :
	(if [ !  -d $(UAL)/ext/UI/include ]; \
		then mkdir  $(UAL)/ext/UI/include; fi;)
	(if [ !  -d $(UAL)/ext/UI/include/UAL ]; \
		then mkdir  $(UAL)/ext/UI/include/UAL; fi;)
	(if [ !  -d $(UAL)/ext/UI/lib ]; \
		then mkdir  $(UAL)/ext/UI/lib; fi;)	
	(if [ !  -d $(UAL)/ext/UI/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL)/ext/UI/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL)/ext/UI/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL)/ext/UI/lib/$(UAL_ARCH)/obj; fi;)	
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL)/ext/UI/include/UAL
	rm -rf $(UAL)/ext/UI/lib/$(UAL_ARCH)
