include $(UAL)/env/$(UAL_ARCH)/Makefile.config

DIRS := ./PAC ./Survey ./Optics ./SMF 


compile :
	(if [ !  -d $(UAL_PAC)/include ]; then mkdir  $(UAL_PAC)/include; fi;)
	(if [ !  -d $(UAL_PAC)/lib ]; \
		then mkdir  $(UAL_PAC)/lib; fi;)	
	(if [ !  -d $(UAL_PAC)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_PAC)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_PAC)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_PAC)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done


clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_PAC)/include
	rm -rf $(UAL_PAC)/lib/$(UAL_ARCH)