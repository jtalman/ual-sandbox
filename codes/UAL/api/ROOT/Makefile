include $(UAL)/env/$(UAL_ARCH)/Makefile.config

ROOTCFLAGS    = $(shell root-config --cflags)
ROOTLIBS      = $(shell root-config --libs)
ROOTGLIBS     = $(shell root-config --glibs)
INC += -I$(UAL_CORE)/include $(ROOTCFLAGS)

CCFLAGS += -Wno-long-long

LIBDIR = $(UAL_CORE)/lib/$(UAL_ARCH)
OBJDIR = $(LIBDIR)/obj

LIBS   = -L$(UAL_CORE)/lib/$(UAL_ARCH) -lUal

# Header files of classes included in the ROOT dictionary

HEADERS:=
include Headers

# Dictionary object

DICT := $(OBJDIR)/UALDict.o

compile : $(LIBDIR)/libRootUalCore.so 
	rm -f *Dict.*
	@echo "libUalRoot.so Needs to be remade."

clean:
	rm -rf $(LIBDIR)/libRootUalCore.so 
	rm -f $(DICT)
	rm -f *Dict.*

$(LIBDIR)/libRootUalCore.so: $(DICT)
	$(DLD) $(DLDFLAGS) -o $@  $< $(LIBS) $(ROOTLIBS) $(ROOTGLIBS)


$(OBJDIR)/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC)  -c $< -o $@;


UALDict.cc: $(HEADERS)  LinkDef.h
	@echo "Generating dictionary $@..."
	$(ROOTSYS)/bin/rootcint -f $@ -c -I$(UAL_CORE)/include  $^

