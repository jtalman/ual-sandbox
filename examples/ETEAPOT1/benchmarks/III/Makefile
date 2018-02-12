LD       = g++                       # Linker
LDFLAGS  =                           # Linker flags
CC       = $(LD)
CCFLAGS  = $(LDFLAGS) -fpermissive

INC  =
INC += -I. -I./src
INC += -I$(UAL)/ext/UI/include
INC += -I$(UAL_SXF)/src
INC += -I$(UAL_ACCSIM)/src
INC += -I$(UAL_TEAPOT)/src
INC += -I$(UAL_ETEAPOT)/src
INC += -I$(UAL_ETEAPOT_MltTurn)/src
INC += -I$(UAL_PAC)/src
INC += -I$(UAL)/codes/UAL/include
INC += -I$(UAL_ZLIB)/src
INC += -I$(UAL)/tools/include
INC += -I$(SXF)/src
INC += -I/home/ual/tools/gsl/include

LIBS  =
LIBS += -L$(UAL)/ext/UI/lib/$(UAL_ARCH)          -lUalUI
LIBS += -L$(UAL_TIBETAN)/lib/$(UAL_ARCH)         -lTibetan
LIBS += -L$(UAL_SXF)/lib/$(UAL_ARCH)             -lUalSXF2
LIBS += -L$(UAL_ACCSIM)/lib/$(UAL_ARCH)          -lAccsim
LIBS += -L$(UAL_TEAPOT)/lib/$(UAL_ARCH)          -lTeapot
LIBS += -L$(UAL_ETEAPOT)/lib/$(UAL_ARCH)         -lETeapot
LIBS += -L$(UAL_ETEAPOT_MltTurn)/lib/$(UAL_ARCH) -lETEAPOT_MltTurn
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH)             -lPacSMF -lPacOptics -lPac
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH)            -lZTps
LIBS += -L$(UAL)/codes/UAL/lib/$(UAL_ARCH)       -lUal
LIBS += -L$(SXF)/lib/$(UAL_ARCH)                 -lSXF
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH)           -lpcre -lxml2 -lrfftw -lfftw
LIBS +=                                          -lm 

SRC_DIR  = .
OBJ_DIR  = ./lib/$(UAL_ARCH)/obj

OBJS = $(OBJ_DIR)/positionPrinter.o

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	@echo object file compilation
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@

compile: $(OBJS) ETEAPOT_MltTurn positionPrinter

./positionPrinter: $(OBJ_DIR)/positionPrinter.o $(OBJS)
	$(CC) -o $@  $(CCFLAGS) $(INC) $(OBJ_DIR)/positionPrinter.o $(OBJS) $(LIBS)

./ETEAPOT_MltTurn: $(OBJ_DIR)/ETEAPOT_MltTurn.o $(OBJS)
	$(CC) -o $@  $(CCFLAGS) $(INC) $(OBJ_DIR)/ETEAPOT_MltTurn.o $(OBJS) $(LIBS)

clean:
	rm -f ETEAPOT_MltTurn ETEAPOT_MltTurn.cc $(OBJ_DIR)/*.o SplitNikolayOut.pl userBunch xmgracePrint.hh positionPrinter.cc positionPrinter.hh simulatedProbeValues verifyMltTrackerSpin verifyDipoleTrackerSpin verifyMarkerTrackerSpin spin printProbeValues extractParameters.h setBeamAttributes.hh designBeamValues.hh IG NikolayOut initialSpin OUT echo.sxf m_m out/cpp/* out/TWISS/* out/VERIF/*
