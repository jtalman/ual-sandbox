include $(UAL_ACCSIM)/src/Makefile.config

ROOTCFLAGS    = $(shell root-config --cflags)
ROOTLIBS      = $(shell root-config --libs)
ROOTGLIBS     = $(shell root-config --glibs)

INC += -I$(UAL_ACCSIM)/include $(ROOTCFLAGS)

CCFLAGS += -Wno-long-long

LIBDIR = $(UAL_ACCSIM)/lib/$(UAL_ARCH)
OBJDIR = $(LIBDIR)/obj

LIBS   = -L$(UAL_ACCSIM)/lib/$(UAL_ARCH) -lAccsim  $(ROOTGLIBS)

# Header files of classes included in the ROOT dictionary

HEADERS:=
include Headers

# Dictionary object

DICT := $(OBJDIR)/ACCSIMDict.o 

compile : $(LIBDIR)/libRootAccsim.so 
	rm -f *Dict.*
	@echo "libUalRoot.so Needs to be remade."

clean:
	rm -rf $(LIBDIR)/libRootAccsim.so 
	rm -f $(DICT)
	rm -f *Dict.*

cleanUALROOT:
	make clean

$(LIBDIR)/libRootAccsim.so: $(DICT)
	$(DLD) $(DLDFLAGS) -o $@  $< $(LIBS) 

#$(OBJDIR)/%.o : %.cc
#	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;


ACCSIMDict.cc: $(HEADERS)  LinkDef.h
	@echo "Generating dictionary $@..."
	$(ROOTSYS)/bin/rootcint -f $@ -c -I$(UAL_PAC)/include -I$(UAL_CORE)/include -I$(UAL_ZLIB)/src -I$(UAL_ACCSIM)/src  $^

