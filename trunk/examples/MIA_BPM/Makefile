CC       = g++                                    # Compiler
CCFLAGS  = -O -fpic -ansi -Wall         # Compilation flags

DLD      = g++                       # Dynamic linker
DLDFLAGS = -shared                   # Dynamic linker flags
LD       = g++                       # Linker
LDFLAGS  =                           # Linker flags

INC += -I. -I./src
INC += -I$(UAL_SXF)/src 
INC += -I$(UAL_ACCSIM)/src 
INC += -I$(UAL_TEAPOT)/src 
INC += -I$(UAL_PAC)/src 
INC += -I$(UAL)/codes/UAL/include
INC += -I$(UAL_ZLIB)/src
INC += -I$(UAL)/tools/include 
INC += -I$(SXF)/src 
 
LIBS += -L$(UAL_TIBETAN)/lib/$(UAL_ARCH)  	-lTibetan
LIBS += -L$(UAL_SXF)/lib/$(UAL_ARCH) 		-lUalSXF
LIBS += -L$(UAL_ACCSIM)/lib/$(UAL_ARCH) 	-lAccsim 
LIBS += -L$(UAL_TEAPOT)/lib/$(UAL_ARCH) 	-lTeapot 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH)  		-lPacSMF -lPacOptics -lPac
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) 		-lZTps
LIBS += -L$(UAL)/codes/UAL/lib/$(UAL_ARCH) 	-lUal
LIBS += -L$(SXF)/lib/$(UAL_ARCH) 		-lSXF
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH) 		-lpcre -lxml2
LIBS += 					-lm

OBJS = 	$(UAL_ARCH)/obj/VBpm.o \
	$(UAL_ARCH)/obj/HBpm.o \
	$(UAL_ARCH)/obj/BPM.o \
	$(UAL_ARCH)/obj/BPMCollector.o \
	$(UAL_ARCH)/obj/timer.o	

compile : ./$(UAL_ARCH)/tracker ./$(UAL_ARCH)/bad_bpm ./$(UAL_ARCH)/noise_all_bpms ./$(UAL_ARCH)/phadv

clean :
	rm -rf ./$(UAL_ARCH)

$(UAL_ARCH)/bad_bpm  :  $(UAL_ARCH)/obj/bad_bpm.o $(OBJS) 
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJS) $(UAL_ARCH)/obj/bad_bpm.o $(LIBS)
$(UAL_ARCH)/noise_all_bpms  :  $(UAL_ARCH)/obj/noise_all_bpms.o $(OBJS) 
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJS) $(UAL_ARCH)/obj/noise_all_bpms.o $(LIBS)
$(UAL_ARCH)/tracker  :  $(UAL_ARCH)/obj/tracker.o $(OBJS) 
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJS) $(UAL_ARCH)/obj/tracker.o $(LIBS)
$(UAL_ARCH)/phadv  :  $(UAL_ARCH)/obj/phadv.o $(OBJS) 
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJS) $(UAL_ARCH)/obj/phadv.o $(LIBS)

$(UAL_ARCH)/obj/tracker.o  :  ./src/tracker.cc
	if [ ! -d ./$(UAL_ARCH) ] ; then (mkdir ./$(UAL_ARCH); mkdir ./$(UAL_ARCH)/out;); fi;
	if [ ! -d ./$(UAL_ARCH)/obj ] ; then (mkdir ./$(UAL_ARCH)/obj;); fi;	
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $< 
$(UAL_ARCH)/obj/bad_bpm.o  :  ./src/bad_bpm.cc
	if [ ! -d ./$(UAL_ARCH) ] ; then (mkdir ./$(UAL_ARCH); mkdir ./$(UAL_ARCH)/out;); fi;
	if [ ! -d ./$(UAL_ARCH)/obj ] ; then (mkdir ./$(UAL_ARCH)/obj;); fi;	
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $< 
$(UAL_ARCH)/obj/noise_all_bpms.o  :  ./src/noise_all_bpms.cc
	if [ ! -d ./$(UAL_ARCH) ] ; then (mkdir ./$(UAL_ARCH); mkdir ./$(UAL_ARCH)/out;); fi;
	if [ ! -d ./$(UAL_ARCH)/obj ] ; then (mkdir ./$(UAL_ARCH)/obj;); fi;	
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $< 
$(UAL_ARCH)/obj/phadv.o  :  ./src/phadv.cc
	if [ ! -d ./$(UAL_ARCH) ] ; then (mkdir ./$(UAL_ARCH); mkdir ./$(UAL_ARCH)/out;); fi;
	if [ ! -d ./$(UAL_ARCH)/obj ] ; then (mkdir ./$(UAL_ARCH)/obj;); fi;	
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $< 

$(UAL_ARCH)/obj/VBpm.o  :  ./src/VBpm.cc
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $<  
$(UAL_ARCH)/obj/HBpm.o  :  ./src/HBpm.cc
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $<  
$(UAL_ARCH)/obj/BPM.o  :  ./src/BPM.cc
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $<  
$(UAL_ARCH)/obj/BPMCollector.o  :  ./src/BPMCollector.cc
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $< 
$(UAL_ARCH)/obj/timer.o  :  ./src/timer.cc
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $<  
