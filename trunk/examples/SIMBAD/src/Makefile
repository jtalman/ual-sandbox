
CC       = g++                       # Compiler
CCFLAGS  = -O -fpic -ansi -Wall      # Compilation flags

DLD      = g++                       # Dynamic linker
DLDFLAGS = -shared                   # Dynamic linker flags
LD       = g++                       # Linker
LDFLAGS  =                           # Linker flags

INC += -I. -I./src
INC += -I$(UAL_SXF)/src 
INC += -I$(UAL_SIMBAD)/src 
INC += -I$(UAL_ACCSIM)/src 
INC += -I$(UAL_TEAPOT)/src 
INC += -I$(UAL_PAC)/src 
INC += -I$(UAL)/codes/UAL/include
INC += -I$(UAL_ZLIB)/src
INC += -I$(UAL)/tools/include 
INC += -I$(SXF)/src 
INC += -I/home/ual/tools/gsl/include
 
LIBS += -L$(UAL_SIMBAD)/lib/$(UAL_ARCH)  	-lSimbad
LIBS += -L$(UAL_TIBETAN)/lib/$(UAL_ARCH)  	-lTibetan
LIBS += -L$(UAL_SXF)/lib/$(UAL_ARCH) 		-lUalSXF -lUalSXF2
LIBS += -L$(UAL_ACCSIM)/lib/$(UAL_ARCH) 	-lAccsim 
LIBS += -L$(UAL_TEAPOT)/lib/$(UAL_ARCH) 	-lTeapot 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH)  		-lPacSMF -lPacOptics -lPac
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) 		-lZTps
LIBS += -L$(UAL)/codes/UAL/lib/$(UAL_ARCH) 	-lUal
LIBS += -L$(SXF)/lib/$(UAL_ARCH) 		-lSXF
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH) 		-lpcre -lxml2 -lrfftw -lfftw
LIBS += 					-lm

SRC_DIR  = .
OBJ_DIR  = ../lib/$(UAL_ARCH)/obj

OBJS =  $(OBJ_DIR)/timer.o \
	$(OBJ_DIR)/Argument.o \
	$(OBJ_DIR)/Arguments.o \
	$(OBJ_DIR)/Shell.o 


compile : $(OBJS) ../$(UAL_ARCH)/fplot 

clean :
	rm -f $(OBJS) ../$(UAL_ARCH)/fplot

../$(UAL_ARCH)/fplot  : $(OBJ_DIR)/fplot.o $(OBJS) 
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJ_DIR)/fplot.o $(OBJS) $(LIBS)

../$(UAL_ARCH)/tracker  : $(OBJ_DIR)/tracker.o $(OBJS) 
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJ_DIR)/tracker.o $(OBJS) $(LIBS)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@

