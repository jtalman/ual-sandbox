include $(UAL_SIMBAD)/src/Makefile.config

ROOTCFLAGS    = $(shell root-config --cflags)
ROOTLIBS      = $(shell root-config --libs)
ROOTGLIBS     = $(shell root-config --glibs)

INC += -I$(UAL_SIMBAD)/include $(ROOTCFLAGS)

CCFLAGS += -Wno-long-long

LIBDIR = $(UAL_SIMBAD)/lib/$(UAL_ARCH)
OBJDIR = $(LIBDIR)/obj

LIBS   = -L$(UAL_SIMBAD)/lib/$(UAL_ARCH) -lSimbad  $(ROOTGLIBS)

# Header files of classes included in the ROOT dictionary

HEADERS:=
include Headers

# Dictionary object

DICT := $(OBJDIR)/SIMBADDict.o 

compile : $(LIBDIR)/libRootSimbad.so 
	rm -f *Dict.*
	@echo "libUalRoot.so Needs to be remade."

clean:
	rm -rf $(LIBDIR)/libRootSimbad.so 
	rm -f $(DICT)
	rm -f *Dict.*

cleanUALROOT:
	make clean

$(LIBDIR)/libRootSimbad.so: $(DICT)
	$(DLD) $(DLDFLAGS) -o $@  $< $(LIBS) 

#$(OBJDIR)/%.o : %.cc
#	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;


SIMBADDict.cc: $(HEADERS)  LinkDef.h
	@echo "Generating dictionary $@..."
	$(ROOTSYS)/bin/rootcint -f $@ -c -I$(UAL_PAC)/include -I$(UAL_CORE)/include -I$(UAL_ZLIB)/src -I$(UAL_SIMBAD)/src  $^

