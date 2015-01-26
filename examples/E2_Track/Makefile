CC       = g++                       # Compiler
CCFLAGS  = -g -fpic -ansi -Wall -Wextra -fpermissive     # Compilation flags

DLD      = g++                       # Dynamic linker
DLDFLAGS = -shared                   # Dynamic linker flags
LD       = g++                       # Linker
LDFLAGS  =                           # Linker flags

INC += -I. -I./src
INC += -I$(UAL)/ext/UI/include
INC += -I$(UAL_SXF)/src
INC += -I$(UAL_ACCSIM)/src
INC += -I$(UAL_TEAPOT)/src
INC += -I$(UAL_ETEAPOT)/src
INC += -I$(UAL_ETEAPOT_MltTurn)/src
INC += -I$(UAL_ETEAPOT2)/src
INC += -I$(UAL_PAC)/src
INC += -I$(UAL)/codes/UAL/include
INC += -I$(UAL_ZLIB)/src
INC += -I$(UAL)/tools/include
INC += -I$(SXF)/src
INC += -I/home/ual/tools/gsl/include

LIBS += -L$(UAL)/ext/UI/lib/$(UAL_ARCH)  	 -lUalUI
LIBS += -L$(UAL_TIBETAN)/lib/$(UAL_ARCH)  	 -lTibetan
LIBS += -L$(UAL_SXF)/lib/$(UAL_ARCH) 		 -lUalSXF2
LIBS += -L$(UAL_ACCSIM)/lib/$(UAL_ARCH) 	 -lAccsim
LIBS += -L$(UAL_TEAPOT)/lib/$(UAL_ARCH) 	 -lTeapot
LIBS += -L$(UAL_ETEAPOT)/lib/$(UAL_ARCH) 	 -lETeapot
LIBS += -L$(UAL_ETEAPOT_MltTurn)/lib/$(UAL_ARCH) -lETEAPOT_MltTurn
LIBS += -L$(UAL_ETEAPOT2)/lib/$(UAL_ARCH)        -lETeapot2
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH)  		 -lPacSMF -lPacOptics -lPac
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) 		 -lZTps
LIBS += -L$(UAL)/codes/UAL/lib/$(UAL_ARCH) 	 -lUal
LIBS += -L$(SXF)/lib/$(UAL_ARCH) 		 -lSXF
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH) 		 -lpcre -lxml2 -lrfftw -lfftw
LIBS += 					 -lm

SRC_DIR  = .
OBJ_DIR  = ./lib/$(UAL_ARCH)/obj

 OBJS =
#OBJS = $(OBJ_DIR)/positionPrinter.o

compile : $(OBJS) ./E2_Track

clean :
	rm -f $(OBJS) $(OBJ_DIR)/E2_Track.o ./E2_Track
	rm -rf m_m echo.sxf betaFunctions* OUT* initialSpin* NikolayOut* IG* out/* out/cpp/* out/STT/* out/TWISS/* out/VERIF/*
	mkdir out/cpp

./E2_Track: $(OBJ_DIR)/E2_Track.o
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJ_DIR)/E2_Track.o $(OBJS) $(LIBS)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@

