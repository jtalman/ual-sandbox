include $(UAL_GUI)/COIN3D/src/Makefile.config

MY_DIR  := $(UAL_GUI)/COIN3D/
DIRS = ./Viewers

OBJS = 

include ./Viewers/Objects

compile : 
	(if [ !  -d $(MY_DIR)/include/UAL/COIN3D ]; \
		then mkdir  $(MY_DIR)/include/UAL/COIN3D; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(DLD) $(DLDFLAGS) -o $(MY_DIR)/lib/$(UAL_ARCH)/libUalCoin3DViewers.so $(OBJS) $(LIBS)


clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(MY_DIR)/include/UAL/COIN3D
	rm -f $(MY_DIR)/lib/$(UAL_ARCH)/libUalCoin3DViewers.so
