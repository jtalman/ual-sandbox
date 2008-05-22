include $(UAL)/env/$(UAL_ARCH)/Makefile.config

ROOTCFLAGS    = $(shell root-config --cflags)
INC +=  -I$(UAL_AIM)/include \
	-I$(UAL_PAC)/include \
	-I$(UAL_ZLIB)/src \
	-I$(UAL_CORE)/include \
	$(ROOTCFLAGS)

CCFLAGS += -Wno-long-long

LIBDIR = $(UAL_AIM)/lib/$(UAL_ARCH)
OBJDIR = $(LIBDIR)/obj

LIBS = 	-L$(UAL_AIM)/lib/$(UAL_ARCH) -lAim \
	-L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac \
	-L$(UAL_CORE)/lib/$(UAL_ARCH) -lUal

# Header files of classes included in the ROOT dictionary

HEADERS:=
include Headers

# Dictionary object

DICT := $(OBJDIR)/AIMDict.o

compile : $(LIBDIR)/libRootAim.so 
	rm -f *Dict.*
	@echo "libUalRoot.so Needs to be remade."

clean:
	rm -rf $(LIBDIR)
	rm -f $(DICT)
	rm -f *Dict.*

$(LIBDIR)/libRootAim.so: $(DICT)
	$(DLD) $(DLDFLAGS) -o $@  $< $(LIBS) 


$(OBJDIR)/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC)  -c $< -o $@;


AIMDict.cc: $(HEADERS)  LinkDef.h
	@echo "Generating dictionary $@..."
	$(ROOTSYS)/bin/rootcint -f $@ -c $(INC)  $^

