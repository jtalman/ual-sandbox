CC       = g++                       # Compiler
CCFLAGS  = -g -fpic -ansi -Wall      # Compilation flags

LD       = g++                       # Linker

INC += -I.
INC += -I$(UAL)/ext/UI/include
INC += -I$(UAL_SXF)/src
INC += -I$(UAL_TEAPOT)/src
INC += -I$(UAL_PAC)/src
INC += -I$(UAL)/codes/UAL/include
INC += -I$(UAL_ZLIB)/src
INC += -I$(UAL)/tools/include
INC += -I$(SXF)/src

LIBS += -L$(UAL)/ext/UI/lib/$(UAL_ARCH) -lUalUI
LIBS += -L$(UAL_SXF)/lib/$(UAL_ARCH) -lUalSXF2
LIBS += -L$(UAL_TEAPOT)/lib/$(UAL_ARCH) -lTeapot
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacSMF -lPacOptics -lPac
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) -lZTps
LIBS += -L$(UAL)/codes/UAL/lib/$(UAL_ARCH) -lUal
LIBS += -L$(SXF)/lib/$(UAL_ARCH) -lSXF
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH) -lpcre -lxml2 -lrfftw -lfftw
LIBS += -lm

SRC_DIR  = .
OBJ_DIR  = ./lib/$(UAL_ARCH)/obj

EXTERNAL = $(OBJ_DIR)/timer.o \
           $(OBJ_DIR)/positionPrinter.o \
           $(OBJ_DIR)/driftOrDefault.o \
           $(OBJ_DIR)/rfCavity.o

THINSPIN =  $(OBJ_DIR)/threeVector.o \
	$(OBJ_DIR)/fourVector.o \
	$(OBJ_DIR)/bendAndMultipole.o \
	$(OBJ_DIR)/PropagatorSequence.o \
	$(OBJ_DIR)/tracker.o

compile : $(THINSPIN) ./tracker

clean :
	rm -f $(EXTERNAL) $(THINSPIN) ./tracker

./tracker  : $(EXTERNAL) $(THINSPIN)
	$(LD) -o $@  $(LDFLAGS) $(INC) $(EXTERNAL) $(THINSPIN) $(LIBS)


$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@

external : $(EXTERNAL)
