include $(UAL)/env/$(UAL_ARCH)/Makefile.config

ROOTCFLAGS    = $(shell root-config --cflags)
ROOTLIBS      = $(shell root-config --libs)
ROOTGLIBS     = $(shell root-config --glibs)

INC      = -I$(UAL_TEAPOT)/src -I$(UAL_PAC)/src  -I$(UAL_CORE)/include -I$(UAL_ZLIB)/src -I$(ROOTCFLAGS)
LIBS     = -L$(UAL_TEAPOT)/lib/$(UAL_ARCH) -L$(UAL_PAC)/lib/$(UAL_ARCH) \
	 	-L$(UAL_ZLIB)/lib/$(UAL_ARCH) \
		-L$(UAL_CORE)/lib/$(UAL_ARCH) -lUal -ldl 



CCFLAGS += -Wno-long-long

LIBDIR = $(UAL_TEAPOT)/lib/$(UAL_ARCH)
OBJDIR = $(LIBDIR)/obj

LIBS   = -L$(UAL_TEAPOT)/lib/$(UAL_ARCH) -lTeapot
LIBS  += $(ROOTLIBS) $(ROOTGLIBS)

# Header files of classes included in the ROOT dictionary

HEADERS:=
include Headers

# Dictionary object

DICT := $(OBJDIR)/TEAPOTDict.o 

compile : $(LIBDIR)/libRootTeapot.so 
	rm -f *Dict.*
	@echo "libUalRoot.so Needs to be remade."

clean:
	rm -rf $(LIBDIR)/libRootTeapot.so 
	rm -f $(DICT)
	rm -f *Dict.*

cleanUALROOT:
	make clean

$(LIBDIR)/libRootTeapot.so: $(DICT)
	$(DLD) $(DLDFLAGS) -o $@  $< $(LIBS) 

$(OBJDIR)/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;


TEAPOTDict.cc: $(HEADERS)  LinkDef.h
	@echo "Generating dictionary $@..."
	$(ROOTSYS)/bin/rootcint -f $@ -c -I$(UAL_CORE)/include -I$(UAL_PAC)/include -I$(UAL_TEAPOT)/include  $^

