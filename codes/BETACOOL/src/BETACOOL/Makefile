CC       = g++ -g                      # Compiler
CCFLAGS  = -O -fpic -ansi -Wall      # Compilation flags

DLD      = g++                       # Dynamic linker
DLDFLAGS = -shared                   # Dynamic linker flags
LD       = g++                       # Linker
LDFLAGS  =                           # Linker flags

INC += -I. -I../../src
INC += -I$(BETACOOL)/include 
INC += -I$(UAL_SIMBAD)/src 
INC += -I$(UAL_ACCSIM)/src 
INC += -I$(UAL_TEAPOT)/src 
INC += -I$(UAL_PAC)/src 
INC += -I$(UAL)/codes/UAL/include
INC += -I$(UAL_ZLIB)/src
INC += -I$(UAL)/tools/include 
INC += -I$(SXF)/src 
INC += -I/home/ual/tools/gsl/include
 
 
LIBS += -L$(BETACOOL)/lib/linux         	-lBetacool
LIBS += -L$(UAL_SIMBAD)/lib/$(UAL_ARCH)  	-lSimbad
LIBS += -L$(UAL_TIBETAN)/lib/$(UAL_ARCH)  	-lTibetan
LIBS += -L$(UAL_ACCSIM)/lib/$(UAL_ARCH) 	-lAccsim 
LIBS += -L$(UAL_TEAPOT)/lib/$(UAL_ARCH) 	-lTeapot 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH)  		-lPacSMF -lPacOptics -lPac
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) 		-lZTps
LIBS += -L$(UAL)/codes/UAL/lib/$(UAL_ARCH) 	-lUal
LIBS += -L$(SXF)/lib/$(UAL_ARCH) 		-lSXF
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH) 		-lpcre -lxml2 -lrfftw -lfftw
LIBS += 					-lm

SRC_DIR  = .
OBJ_DIR  = $(UAL_BETACOOL)/lib/$(UAL_ARCH)/obj

OBJS =  $(OBJ_DIR)/Ring.o \
	$(OBJ_DIR)/CompositeTracker.o

compile : $(UAL_BETACOOL)/lib/$(UAL_ARCH)/libUalBetacool.so
	if [ ! -d $(UAL_BETACOOL)/include/ ] ; then mkdir $(UAL_BETACOOL)/include/ ; fi;
	if [ ! -d $(UAL_BETACOOL)/include/BETACOOL/ ] ; then mkdir $(UAL_BETACOOL)/include/BETACOOL; fi;
	cp *.hh $(UAL_BETACOOL)/include/BETACOOL

clean :
	rm -f $(OBJS) $(UAL_BETACOOL)/lib/$(UAL_ARCH)/libUalBetacool.so

$(UAL_BETACOOL)/lib/$(UAL_ARCH)/libUalBetacool.so: $(OBJS)
	$(DLD) $(DLDFLAGS) -o $(UAL_BETACOOL)/lib/$(UAL_ARCH)/libUalBetacool.so $(OBJS) $(LIBS)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@

