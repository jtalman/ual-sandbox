CC       = g++                       # Compiler
CCFLAGS  = -O -fpic -ansi -Wall      # Compilation flags

DLD      = g++                       # Dynamic linker
DLDFLAGS = -shared                   # Dynamic linker flags
LD       = g++                       # Linker
LDFLAGS  =                           # Linker flags

INC += -I. 
INC += -I$(UAL_ICE)/src 
INC += -I$(UAL_PAC)/src 
INC += -I$(UAL)/codes/UAL/include
INC += -I$(UAL_ZLIB)/src
INC += -I$(UAL)/tools/include 

LIBS += -L$(UAL_ICE)/lib/$(UAL_ARCH) 		-lIce
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH)  		-lPacSMF -lPacOptics -lPac
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) 		-lZTps
LIBS += -L$(UAL)/codes/UAL/lib/$(UAL_ARCH) 	-lUal
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH) 		-lpcre 
LIBS += -L$(XERCESCROOT)/lib 			-lxerces-c 
LIBS += 					-lm

compile : ./$(UAL_ARCH)/puls_single_bunch ./$(UAL_ARCH)/puls_4_bunch

clean :
	rm -rf ./$(UAL_ARCH)

$(UAL_ARCH)/puls_single_bunch  :  $(UAL_ARCH)/obj/puls_single_bunch.o  
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJS) $(UAL_ARCH)/obj/puls_single_bunch.o $(LIBS)
$(UAL_ARCH)/obj/puls_single_bunch.o  :  ./src/puls_single_bunch.cc
	if [ ! -d ./$(UAL_ARCH) ] ; then (mkdir ./$(UAL_ARCH); mkdir ./$(UAL_ARCH)/out;); fi;
	if [ ! -d ./$(UAL_ARCH)/obj ] ; then (mkdir ./$(UAL_ARCH)/obj;); fi;	
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $< 

$(UAL_ARCH)/puls_4_bunch  :  $(UAL_ARCH)/obj/puls_4_bunch.o  
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJS) $(UAL_ARCH)/obj/puls_4_bunch.o $(LIBS)
$(UAL_ARCH)/obj/puls_4_bunch.o  :  ./src/puls_4_bunch.cc
	if [ ! -d ./$(UAL_ARCH) ] ; then (mkdir ./$(UAL_ARCH); mkdir ./$(UAL_ARCH)/out;); fi;
	if [ ! -d ./$(UAL_ARCH)/obj ] ; then (mkdir ./$(UAL_ARCH)/obj;); fi;	
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $< 

