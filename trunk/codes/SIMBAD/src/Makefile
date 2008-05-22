include $(UAL_SIMBAD)/src/Makefile.config

DIRS := ./SIMBAD


compile :
	(if [ !  -d $(UAL_SIMBAD)/include ]; then mkdir  $(UAL_SIMBAD)/include; fi;)
	(if [ !  -d $(UAL_SIMBAD)/lib ]; \
		then mkdir  $(UAL_SIMBAD)/lib; fi;)	
	(if [ !  -d $(UAL_SIMBAD)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(UAL_SIMBAD)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(UAL_SIMBAD)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(UAL_SIMBAD)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done


clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_SIMBAD)/include
	rm -rf $(UAL_SIMBAD)/lib/$(UAL_ARCH)
