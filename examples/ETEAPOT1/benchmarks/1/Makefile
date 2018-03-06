LD       = g++                       # Linker
LDFLAGS  =                           # Linker flags
CC       = $(LD)
CCFLAGS  = $(LDFLAGS) -g

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

all: $(OBJ_DIR)/y.tab.o $(OBJ_DIR)/determineTwiss.o ./determineTwiss

$(OBJ_DIR)/y.tab.o : $(SRC_DIR)/y.tab.cc
	@echo ./lib/linux/obj/y.tab.o file compilation
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@

$(OBJ_DIR)/determineTwiss.o : $(SRC_DIR)/determineTwiss.cc
	@echo ./lib/linux/obj/determineTwiss.o file compilation
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@

#$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cc
#	@echo object file compilation
#	$(CC) $(CCFLAGS) $(INC) -c $< -o $@

./determineTwiss  : $(OBJ_DIR)/determineTwiss.o $(OBJS)
	@echo OBJS $(OBJS)
	$(CC) -o $@  $(CCFLAGS) $(INC) $(OBJ_DIR)/y.tab.o $(OBJ_DIR)/determineTwiss.o $(OBJS) $(LIBS)

clean :
	rm -f out/TWISS/StndrdPrtcl* determineTwiss y.tab initialSpin m_m OUT echo.sxf betaFunctions try.log *.hh *.h printProbeValues simulatedProbeValues trtrin trtrout lib/linux/obj/*.o out/cpp/*
