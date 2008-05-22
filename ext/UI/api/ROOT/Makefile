include $(UAL)/ext/UI/src/Makefile.config

ROOTCFLAGS    = $(shell root-config --cflags)
ROOTLIBS      = $(shell root-config --libs)
ROOTGLIBS     = $(shell root-config --glibs)

INC += $(ROOTCFLAGS)

CCFLAGS += -Wno-long-long

LIBDIR = $(UAL)/ext/UI/lib/$(UAL_ARCH)
OBJDIR = $(LIBDIR)/obj


# Header files of classes included in the ROOT dictionary

HEADERS:=
include Headers

# Dictionary object

DICT := $(OBJDIR)/UALDict.o

compile : $(LIBDIR)/libRootUalUI.so 
	# rm -f *Dict.*

clean:
	rm -rf $(LIBDIR)/libRootUalUI.so 
	rm -f $(DICT)
	rm -f *Dict.*

$(LIBDIR)/libRootUalUI.so: $(DICT)
	$(DLD) $(DLDFLAGS) -o $@  $< $(LIBS) $(ROOTLIBS) $(ROOTGLIBS)


$(OBJDIR)/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC)  -c $< -o $@;


UALDict.cc: $(HEADERS)  LinkDef.h
	@echo "Generating dictionary $@..."
	$(ROOTSYS)/bin/rootcint -f $@ -c  \
	-I$(UAL)/ext/UI/include/ \
	-I$(UAL)/codes/UAL/include/  \
	-I$(UAL)/codes/ZLIB/include/ \
	-I$(UAL)/codes/PAC/include/ \
	-I$(UAL)/codes/TIBETAN/include/ \
	$^

