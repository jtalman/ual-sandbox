include $(SXF)/config/$(SXF_ARCH)/Makefile.config

ROOTCFLAGS    = $(shell root-config --cflags)
ROOTLIBS      = $(shell root-config --libs)
ROOTGLIBS     = $(shell root-config --glibs)

INC += $(ROOTCFLAGS)

CCFLAGS += -Wno-long-long

LIBDIR = $(SXF)/lib/$(SXF_ARCH)
OBJDIR = $(LIBDIR)/obj

LIBS   = -L$(SXF)/lib/$(SXF_ARCH) -lEchoSXF -lSXF $(ROOTLIBS) $(ROOTGLIBS)

# Header files of classes included in the ROOT dictionary

SXFDIR     = $(SXF)/src/sxf
ECHOSXFDIR = $(SXF)/src/echo_sxf

SXFHEADERS:=
include SXFHeaders

ECHOSXFHEADERS:=
include Echo_sxfHeaders

# Dictionary objects

SXFDICT := $(OBJDIR)/SXFDict.o
ECHOSXFDICT :=$(OBJDIR)/EchoSXFDict.o

compile : $(LIBDIR)/libRootSXF.so $(LIBDIR)/libRootEchoSXF.so
	rm -f *Dict.*
	@echo "libUalRoot.so Needs to be remade."

clean:
	rm -rf $(LIBDIR)/libRootSXF.so $(LIBDIR)/libRootEchoSXF.so
	rm -f *Dict.*
	rm -f $(OBJDIR)/SXFDict.o $(OBJDIR)/EchoSXFDict.o

$(LIBDIR)/libRootSXF.so: $(SXFDICT)
	$(DLD) $(DLDFLAGS) -o $@  $< $(LIBS) 

$(LIBDIR)/libRootEchoSXF.so: $(ECHOSXFDICT)
	$(DLD) $(DLDFLAGS) -o $@  $< $(LIBS) 

$(OBJDIR)/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;

SXFDict.cc: $(SXFHEADERS)  SXFLinkDef.h
	@echo "Generating dictionary $@..."
	$(ROOTSYS)/bin/rootcint -f $@ -c -I$(SXF)/src  $^

EchoSXFDict.cc:  $(ECHOSXFHEADERS)  Echo_SXFLinkDef.h
	@echo "Generating dictionary $@..."
	$(ROOTSYS)/bin/rootcint -f $@ -c -I$(SXF)/src  $^
