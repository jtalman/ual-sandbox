include $(UAL_GUI)/QT/src/Makefile.config

MY_DIR  := $(UAL_GUI)/QT/

OBJS     =  
include  ./Objects

HHS      = 	Def.hh \
		PlayerShell.hh

UICDECLS = MainPlayerUI.hh \
	   PlayerUI.hh \
	   TablePageUI.hh 

UICIMPLS = MainPlayerUI.cc \
	   PlayerUI.cc \
	   TablePageUI.cc 

SRCMOC   = moc_MainPlayerUI.cc \
	   moc_PlayerUI.cc \
	   moc_BasicPlayer.cc \
           moc_BasicViewer.cc \
           moc_BasicEditor.cc \
	   moc_TablePageUI.cc \
	   moc_BeamEditor.cc

compile: $(OBJS)
	if [ ! -d $(MY_DIR)/include/UAL/QT/Player ] ; \
		then mkdir $(MY_DIR)/include/UAL/QT/Player ; fi;
	cp $(HHS) $(MY_DIR)/include/UAL/QT/Player/.

qt: $(UICDECLS) $(UICIMPLS) $(SRCMOC)

clean:
	rm -rf $(OBJS)
	rm -rf $(MY_DIR)/include/UAL/QT/Player

MainPlayerUI.cc: MainPlayerUI.ui 
	$(UIC) -impl MainPlayerUI.hh MainPlayerUI.ui   -o $@
PlayerUI.cc: PlayerUI.ui 
	$(UIC) -impl PlayerUI.hh PlayerUI.ui   -o $@
TablePageUI.cc: TablePageUI.ui 
	$(UIC) -impl TablePageUI.hh TablePageUI.ui   -o $@







