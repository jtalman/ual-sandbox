CC       = g++                                    # Compiler
CCFLAGS  = -O -fpic -ansi -Wall         # Compilation flags

DLD      = g++                       # Dynamic linker
DLDFLAGS = -shared                   # Dynamic linker flags
LD       = g++                       # Linker
LDFLAGS  =                           # Linker flags

INC += -I. 
INC += -I$(UAL_SXF)/src 
INC += -I$(UAL_TEAPOT)/src 
INC += -I$(UAL_PAC)/src 
INC += -I$(UAL)/codes/UAL/include
INC += -I$(UAL_ZLIB)/src
INC += -I$(UAL)/tools/include -I$(UAL)/tools/include/libxml2 
INC += -I$(SXF)/src   

LIBS += -L$(UAL_SXF)/lib/$(UAL_ARCH) 		-lUalSXF
LIBS += -L$(UAL_TEAPOT)/lib/$(UAL_ARCH) 	-lTeapot 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH)  		-lPacSMF -lPacOptics -lPac
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) 		-lZTps
LIBS += -L$(UAL)/codes/UAL/lib/$(UAL_ARCH) 	-lUal
LIBS += -L$(SXF)/lib/$(UAL_ARCH) 		-lSXF
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH) 		-lpcre -lxml2
LIBS += 					-lm

OBJS = $(UAL_ARCH)/obj/timer.o	

compile : ./$(UAL_ARCH)/evolver ./$(UAL_ARCH)/tracker

clean :
	rm -rf ./$(UAL_ARCH)


$(UAL_ARCH)/evolver  :  $(UAL_ARCH)/obj/evolver.o $(OBJS) 
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJS) $(UAL_ARCH)/obj/evolver.o $(LIBS)
$(UAL_ARCH)/tracker  :  $(UAL_ARCH)/obj/tracker.o $(OBJS) 
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJS) $(UAL_ARCH)/obj/tracker.o $(LIBS)

$(UAL_ARCH)/obj/evolver.o  :  ./src/evolver.cc
	if [ ! -d ./$(UAL_ARCH) ] ; then (mkdir ./$(UAL_ARCH); mkdir ./$(UAL_ARCH)/out;); fi;
	if [ ! -d ./$(UAL_ARCH)/obj ] ; then (mkdir ./$(UAL_ARCH)/obj;); fi;	
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $< 
$(UAL_ARCH)/obj/tracker.o  :  ./src/tracker.cc
	if [ ! -d ./$(UAL_ARCH) ] ; then (mkdir ./$(UAL_ARCH); mkdir ./$(UAL_ARCH)/out;); fi;
	if [ ! -d ./$(UAL_ARCH)/obj ] ; then (mkdir ./$(UAL_ARCH)/obj;); fi;	
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $< 
$(UAL_ARCH)/obj/timer.o  :  ./src/timer.cc
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $<  
