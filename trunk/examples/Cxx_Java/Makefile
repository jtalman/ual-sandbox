
CC       = g++                                    # Compiler
CCFLAGS  = -O -fpic -ansi  -Wall                  # Compilation flags

DLD      = g++                                    # Dynamic linker
DLDFLAGS = -shared                                # Dynamic linker flags
LD       = g++                                    # Linker
LDFLAGS  =                                        # Linker flags

INC += -I. 
INC += -I$(UAL_CORE)/include
INC += -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux 


LIBS += -L$(UAL_CORE)/lib/$(UAL_ARCH) -lUal
LIBS += -L$(JAVA_HOME)/jre/lib/i386 -L$(JAVA_HOME)/jre/lib/i386/client -ljava  -ljvm -lverify
LIBS += -lm

OBJS =  

compile : ./$(UAL_ARCH)/demo 

clean :
	rm -rf ./$(UAL_ARCH)

$(UAL_ARCH)/demo :  $(UAL_ARCH)/obj/demo.o $(OBJS) 
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJS) $(UAL_ARCH)/obj/demo.o $(LIBS)

$(UAL_ARCH)/obj/demo.o  :  ./src/demo.cc
	if [ ! -d ./$(UAL_ARCH) ] ; then (mkdir ./$(UAL_ARCH); mkdir ./$(UAL_ARCH)/out;); fi;
	if [ ! -d ./$(UAL_ARCH)/obj ] ; then (mkdir ./$(UAL_ARCH)/obj;); fi;	
	$(CC) -o $@  $(CCFLAGS) $(INC) -c $< 

