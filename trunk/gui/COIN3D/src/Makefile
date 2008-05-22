include $(UAL_GUI)/COIN3D/src/Makefile.config

MY_DIR  := $(UAL_GUI)/COIN3D/

DIRS    := ./UAL


compile :
	(if [ !  -d $(MY_DIR)/include ]; then mkdir  $(MY_DIR)/include; fi;)
	(if [ !  -d $(MY_DIR)/lib ]; \
		then mkdir  $(MY_DIR)/lib; fi;)	
	(if [ !  -d $(MY_DIR)/lib/$(UAL_ARCH) ]; \
		then mkdir  $(MY_DIR)/lib/$(UAL_ARCH); fi;)
	(if [ !  -d $(MY_DIR)/lib/$(UAL_ARCH)/obj ]; \
		then mkdir  $(MY_DIR)/lib/$(UAL_ARCH)/obj; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done


clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(MY_DIR)/include
	rm -rf $(MY_DIR)/lib/$(UAL_ARCH)
