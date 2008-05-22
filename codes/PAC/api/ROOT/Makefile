include $(UAL)/env/$(UAL_ARCH)/Makefile.config

ROOTCFLAGS    = $(shell root-config --cflags)
ROOTLIBS      = $(shell root-config --libs)
ROOTGLIBS     = $(shell root-config --glibs)

INC += -I$(UAL_CORE)/include -I$(UAL_PAC)/include -I$(UAL_PAC)/src
INC += -I$(UAL_ZLIB)/src $(ROOTCFLAGS)

CCFLAGS += -Wno-long-long

LIBDIR = $(UAL_PAC)/lib/$(UAL_ARCH)
OBJDIR = $(LIBDIR)/obj

LIBS   = -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac -lPacOptics
LIBS  += -L$(UAL_CORE)/lib/$(UAL_ARCH) -lUal 
LIBS  += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) -lZTps $(ROOTLIBS) $(ROOTGLIBS)

# Header files of classes included in the ROOT dictionary

HEADERS:=
include Headers

# Dictionary object

DICT := $(OBJDIR)/PACDict.o 

compile : $(LIBDIR)/libRootPac.so 
	rm -f *Dict.*
	@echo "libUalRoot.so Needs to be remade."

clean:
	rm -rf $(LIBDIR)/libRootPac.so 
	rm -f $(DICT)
	rm -f *Dict.*

$(LIBDIR)/libRootPac.so: $(DICT)
	$(DLD) $(DLDFLAGS) -o $@  $< $(LIBS) 

$(OBJDIR)/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;


PACDict.cc: $(HEADERS)  LinkDef.h
	@echo "Generating dictionary $@..."
	$(ROOTSYS)/bin/rootcint -f $@ -c -p -I$(UAL_CORE)/include -I$(UAL_PAC)/include -I$(UAL_PAC)/src -I$(UAL_ZLIB)/src $^

