ifeq ($(USE_GPU), TRUE)
CC       = nvcc  -arch=sm_20  # Compiler
CCFLAGS  = --compiler-options '-fpic -ansi -Wall -g '      # Compilation flags
DLD      = nvcc                      # Dynamic linker
DLDFLAGS = -shared                   # Dynamic linker flags
LD       = nvcc                       # Linker
else
CC       = g++                       # Compiler                                 
CCFLAGS  = -O -fpic -ansi -Wall      # Compilation flags                        

DLD      = g++                       # Dynamic linker                           
DLDFLAGS = -shared                   # Dynamic linker flags                     
LD       = g++                       # Linker  
endif


LDFLAGS  =                           # Linker flags

INC += -I. -I./src 
INC += -I$(UAL)/ext/UI/include
INC += -I$(UAL_SXF)/src
INC += -I$(UAL_SPINK)/src
INC += -I$(UAL_ACCSIM)/src
INC += -I$(UAL_TEAPOT)/src
INC += -I$(UAL_PAC)/src
INC += -I$(UAL)/codes/UAL/include
INC += -I$(UAL_ZLIB)/src
INC += -I$(UAL)/tools/include
INC += -I$(SXF)/src
INC += -I/home/ual/tools/gsl/include
LIBS += -L$(UAL)/ext/UI/lib/$(UAL_ARCH)  	-lUalUI
LIBS += -L$(UAL_SPINK)/lib/$(UAL_ARCH)  	-lSpink
LIBS += -L$(UAL_TIBETAN)/lib/$(UAL_ARCH)  	-lTibetan
LIBS += -L$(UAL_SXF)/lib/$(UAL_ARCH) 		-lUalSXF2
LIBS += -L$(UAL_ACCSIM)/lib/$(UAL_ARCH) 	-lAccsim
LIBS += -L$(UAL_TEAPOT)/lib/$(UAL_ARCH) 	-lTeapot
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH)  		-lPacSMF -lPacOptics -lPac
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) 		-lZTps
LIBS += -L$(UAL)/codes/UAL/lib/$(UAL_ARCH) 	-lUal
LIBS += -L$(SXF)/lib/$(UAL_ARCH) 		-lSXF
LIBS += -L$(UAL)/tools/lib/$(UAL_ARCH) 		-lpcre -lxml2 -lrfftw -lfftw
LIBS += 					-lm
LIBS +=	-L$(UAL)/tools/lib/linux -lgsl
ifeq ($(USE_GPU), TRUE)
LIBS +=         -L/usr/local/cuda/lib64 -lcudart
#LIBS += -L/usr/local/pgiCDK-11.5/linux86-64/2011/cuda/3.1/lib64 -lcudart
endif

SRC_DIR  = .
OBJ_DIR  = ./lib/$(UAL_ARCH)/obj

OBJS = $(OBJ_DIR)/timer.o \
	   $(OBJ_DIR)/PositionPrinter.o \
	   $(OBJ_DIR)/SpinPrinter.o 
 


compile :  $(OBJS) ./tracker 

ifeq ($(USE_GPU), TRUE) 
compile:	$(OBJS) ./tracker ./gpuStep
endif


clean :
ifeq ($(USE_GPU), TRUE)
	rm -f $(OBJS) $(OBJ_DIR)/gpuStep.o ./gpuStep
endif
	rm -f $(OBJS) $(OBJ_DIR)/tracker.o  ./tracker 

ifeq ($(USE_GPU), TRUE)
./gpuStep  : $(OBJ_DIR)/gpuStep.o $(OBJS)
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJ_DIR)/gpuStep.o $(OBJS) $(LIBS)
endif

./tracker     : $(OBJ_DIR)/tracker.o $(OBJS)
	$(LD) -o $@  $(LDFLAGS) $(INC) $(OBJ_DIR)/tracker.o $(OBJS) $(LIBS)

ifeq ($(USE_GPU), TRUE)
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	$(CC)  $(CCFLAGS) $(INC) -c $< -o $@
endif

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CC)  $(CCFLAGS) $(INC) -c $< -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cc
	$(CC)  $(CCFLAGS) $(INC) -c $< -o $@

