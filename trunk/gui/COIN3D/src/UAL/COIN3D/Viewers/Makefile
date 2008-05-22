include $(UAL_GUI)/COIN3D/src/Makefile.config

MY_DIR  := $(UAL_GUI)/COIN3D/

OBJS     = 

include  ./Objects

HHS      = Def.hh

UICDECLS = 

UICIMPLS = 

SRCMOC   = moc_BunchViewer.cc

compile: $(OBJS)
	if [ ! -d $(MY_DIR)/include/UAL/COIN3D/Viewers ] ; \
		then mkdir $(MY_DIR)/include/UAL/COIN3D/Viewers ; fi;
	cp $(HHS) $(MY_DIR)/include/UAL/COIN3D/Viewers/.

qt: $(UICDECLS) $(UICIMPLS) $(SRCMOC)

clean:
	rm -rf $(OBJS)
	rm -rf $(MY_DIR)/include/UAL/COIN3D/Viewers









