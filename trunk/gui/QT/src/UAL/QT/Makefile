include $(UAL_GUI)/QT/src/Makefile.config

MY_DIR  := $(UAL_GUI)/QT/
DIRS = ./Player

OBJS = 

include ./Player/Objects

compile : 
	(if [ !  -d $(MY_DIR)/include/UAL/QT ]; \
		then mkdir  $(MY_DIR)/include/UAL/QT; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(DLD) $(DLDFLAGS) -o $(MY_DIR)/lib/$(UAL_ARCH)/libUalQt.so $(OBJS) $(LIBS)


clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(MY_DIR)/include/UAL/QT
	rm -f $(MY_DIR)/lib/$(UAL_ARCH)/libUalQt.so
