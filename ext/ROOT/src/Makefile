include $(UAL)/env/$(UAL_ARCH)/Makefile.config

ROOTCFLAGS    = $(shell root-config --cflags)
ROOTLIBS      = $(shell root-config --libs )
ROOTGLIBS     = $(shell root-config --glibs)

CCFLAGS += -Wno-long-long

INC = 	 \
	-I$(UAL_SPINK)/src \
	-I$(UAL_TIBETAN)/src \
	-I$(UAL_ACCSIM)/src \
	-I$(UAL_TEAPOT)/src \
	-I$(UAL_SXF)/src  \
	-I$(UAL_PAC)/src  \
	-I$(UAL_ZLIB)/src \
	-I$(UAL_CORE)/src \
	-I$(UAL)/tools/include  -I$(UAL)/tools/include/libxml2 \
	-I$(SXF)/src  \
	$(ROOTCFLAGS) 

LIBS = 	\
	-L$(UAL_SPINK)/lib/$(UAL_ARCH) 		-lSpink \
	-L$(UAL_TIBETAN)/lib/$(UAL_ARCH) 	-lTibetan \
	-L$(UAL_ACCSIM)/lib/$(UAL_ARCH) 	-lRootAccsim -lAccsim \
	-L$(UAL_TEAPOT)/lib/$(UAL_ARCH) 	-lRootTeapot -lTeapot \
	-L$(UAL_SXF)/lib/$(UAL_ARCH)  		-lUalSXF2 \
	-L$(UAL_PAC)/lib/$(UAL_ARCH) 		-lRootPac -lPac -lPacSurvey -lPacOptics -lPacSMF \
	-L$(UAL_ZLIB)/lib/$(UAL_ARCH) 		-lRootZTps -lZTps \
	-L$(UAL_CORE)/lib/$(UAL_ARCH) 		-lRootUalCore -lUal \
	-L$(UAL)/tools/lib/$(UAL_ARCH) 		-lxml2 \
	-L$(SXF)/lib/$(SXF_ARCH) 		-lRootEchoSXF -lRootSXF -lEchoSXF -lSXF \
	$(ROOTLIBS) \
	$(ROOTGLIBS) \
	-lm    


LIBDIR = $(UAL)/ext/ROOT/lib/$(UAL_ARCH)
OBJDIR = $(LIBDIR)/obj

HEADERS = 	RootShell.hh Converters.h LostTree.hh
OBJECTS = 	$(OBJDIR)/RootShell.o \
		$(OBJDIR)/Converters.o\
		$(OBJDIR)/LostTree.o 

DEPS 	:=      $(patsubst $(OBJDIR)/%.o,%.d,$(OBJECTS))

# Dictionary object

DICT :=  $(OBJDIR)/UalRootDICT.o 

compile : $(LIBDIR)/libUalRoot.so
	rm -f *DICT.*

clean:
	rm -rf $(LIBDIR)/libUalRoot.so	
	rm -rf $(OBJECTS) $(DICT)
	rm -f *DICT.*
	rm -rf *.d 

# The UAL Dictionaries are not cleaned with this makefile, this is left to the 
# Dictionary generation makefiles  The UalRoot Dictionary file IS cleaned though.

$(LIBDIR)/libUalRoot.so: $(OBJECTS)  $(DICT)
	$(DLD) $(DLDFLAGS) -o $@ $(DICT) $(OBJECTS) $(LIBS) 



%.d :%.cc
	$(CXX) -M $(CCFLAGS) $(INC) $< | sed s/\\.o/.d/ > $@

$(OBJDIR)/%.o : %.cc %.d
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@

UalRootDICT.cc: $(HEADERS) LinkDef.h 
	@echo "Generating dictionary $@..."
	$(ROOTSYS)/bin/rootcint -f $@ -c -p $(INC) $^

depend: $(DEPS)
	@echo "Dependancies are now up to date"

-include $(DEPS)
