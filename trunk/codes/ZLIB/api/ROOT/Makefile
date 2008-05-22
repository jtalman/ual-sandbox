include $(UAL)/env/$(UAL_ARCH)/Makefile.config

ROOTCFLAGS    = $(shell root-config --cflags)
ROOTLIBS      = $(shell root-config --libs)
ROOTGLIBS     = $(shell root-config --glibs)

INC += -I$(UAL_ZLIB)/src $(ROOTCFLAGS)

CCFLAGS += -Wno-long-long

LIBDIR = $(UAL_ZLIB)/lib/$(UAL_ARCH)
OBJDIR = $(LIBDIR)/obj

LIBS   = -L$(UAL_ZLIB)/lib/$(UAL_ARCH) -lZTps $(ROOTLIBS) $(ROOTGLIBS)

# Header files of classes included in the ROOT dictionary

HEADERS:=
include Headers

# Dictionary objects

DICT := $(OBJDIR)/ZLIBDict.o

compile : $(LIBDIR)/libRootZTps.so 
	rm -f *Dict.*
	@echo "libUalRoot.so Needs to be remade."

clean:
	rm -rf $(LIBDIR)/libRootZTps.so 	
	rm -f $(DICT)
	rm -f *Dict.*

$(LIBDIR)/libRootZTps.so: $(DICT)
	$(DLD) $(DLDFLAGS) -o $@  $< $(LIBS) 

$(OBJDIR)/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC)  -c $< -o $@;

ZLIBDict.cc: $(HEADERS)  LinkDef.h
	@echo "Generating dictionary $@..."
	$(ROOTSYS)/bin/rootcint -f $@ -c -I$(UAL_ZLIB)/src  $^

