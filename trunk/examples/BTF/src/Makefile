# This makefile will make all of the ROOT classes in the current directory
# It finds all dependancies for each file, compiles the classes, calls
# rootcint to make the dictionary file, compiles that, and links 
# the class and dictionary object files together


ROOTCFLAGS    = $(shell root-config --cflags)
ROOTLIBS      = $(shell root-config --new)
ROOTGLIBS     = $(shell root-config --glibs)

INC  += -I.
INC  += -I$(UAL)/ext/ROOT/src 
INC  += -I$(UAL)/ext/SXF/src 
INC  += -I$(UAL_AIM)/src 
INC  += -I$(UAL_ACCSIM)/src 
INC  += -I$(UAL_TIBETAN)/src 
INC  += -I$(UAL_TEAPOT)/src 
INC  += -I$(UAL_PAC)/src 
INC  += -I$(UAL)/codes/UAL/include
INC  += -I$(UAL_ZLIB)/src
INC  += -I$(UAL)/tools/include  -I$(UAL)/tools/include/libxml2
INC  += -I$(SXF)/src 


LIBS += -L$(UAL)/ext/ROOT/lib/$(UAL_ARCH) 	-lUalRoot
LIBS += -L$(UAL_SXF)/lib/$(UAL_ARCH) 		-lUalSXF
LIBS += -L$(UAL_AIM)/lib/$(UAL_ARCH)  		-lRootAim -lAim
LIBS += -L$(UAL_TIBETAN)/lib/$(UAL_ARCH)  	-lTibetan
LIBS += -L$(UAL_ACCSIM)/lib/$(UAL_ARCH)  	-lAccsim
LIBS += -L$(UAL_TEAPOT)/lib/$(UAL_ARCH) 	-lTeapot 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH)  		-lPacSMF -lPacOptics -lPac
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) 		-lZTps
LIBS += -L$(UAL)/codes/UAL/lib/$(UAL_ARCH) 	-lUal
LIBS += -L$(SXF)/lib/$(UAL_ARCH) 		-lSXF
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH) 		-lpcre -lxml2
LIBS += 					-lm

# Linux with gcc

CXX           = g++
CXXFLAGS      = -O2 -Wall -fPIC $(ROOTCFLAGS) $(INC)
LD            = g++
LDFLAGS       =  $(ROOTLIBS) $(ROOTGLIBS) $(LIBS)
SOFLAGS       = -shared 


OBJS :=  BTFShell.o

# Dictionary object

DICT :=  BTFShellDict.o 


compile:            ../lib/$(UAL_ARCH)/libBTFShell.so

clean:
		@rm -rf ../lib/$(UAL_ARCH)
		@rm -f core *.o BTFShellDict.h

# Dependancies for object files

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $<

#Dependancies for so files

../lib/$(UAL_ARCH)/%.so: $(OBJS) BTFShellDict.o
	if [ ! -d ../lib ] ; then (mkdir ../lib; ); fi;
	if [ ! -d ../lib/$(UAL_ARCH) ] ; then (mkdir ../lib/$(UAL_ARCH)); fi;
	$(LD) $(SOFLAGS) -Wl,-soname,$@ $(LDFLAGS) $^ -o $@


# Dendancies of Dictionary files
%Dict.cc: %.hh 
	@echo "Generating dictionary $@..."
	$(ROOTSYS)/bin/rootcint -f $@ -c $(INC) $^ LinkDef.h

