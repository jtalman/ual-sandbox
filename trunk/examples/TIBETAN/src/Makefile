include $(UAL)/env/$(UAL_ARCH)/Makefile.config

DIRS     = APPS
ROOT_DIR = ..
SRC_DIR  = .

INC += -I. 
INC += -I$(UAL_TIBETAN)/src 
INC += -I$(UAL_SXF)/src 
INC += -I$(UAL_ACCSIM)/src 
INC += -I$(UAL_TEAPOT)/src 
INC += -I$(UAL_PAC)/src 
INC += -I$(UAL_CORE)/include
INC += -I$(UAL_ZLIB)/src
INC += -I$(UAL)/tools/include 
INC += -I$(SXF)/src 
INC += -I$(XERCESCROOT)/include 
INC += -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux 

LIBS += -L$(UAL_TIBETAN)/lib/$(UAL_ARCH)  -lTibetan
LIBS += -L$(UAL_SXF)/lib/$(UAL_ARCH)      -lUalSXF
LIBS += -L$(UAL_ACCSIM)/lib/$(UAL_ARCH)   -lAccsim
LIBS += -L$(UAL_TEAPOT)/lib/$(UAL_ARCH)   -lTeapot 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH)      -lPacSMF -lPacOptics -lPacSurvey -lPac
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH)     -lZTps
LIBS += -L$(UAL_CORE)/lib/$(UAL_ARCH)     -lUal
LIBS += -L$(SXF)/lib/$(UAL_ARCH)          -lSXF
LIBS += -L$(UAL)/tools/$(UAL_ARCH)        -lpcre 
LIBS += -L$(XERCESCROOT)/lib              -lxerces-c 
LIBS += -L$(JAVA_HOME)/jre/lib/i386 -L$(JAVA_HOME)/jre/lib/i386/client -ljava  -ljvm -lverify
LIBS += -lm

APPS_OBJS =  	$(ROOT_DIR)/$(UAL_ARCH)/obj/timer.o \
		$(ROOT_DIR)/$(UAL_ARCH)/obj/Shell.o \
	     	$(ROOT_DIR)/$(UAL_ARCH)/obj/tracker.o 

OBJS = 
include ./APPS/GT/Objects

compile: $(APPS_OBJS)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(LD) -o $(ROOT_DIR)/$(UAL_ARCH)/tracker  $(LDFLAGS) $(INC) $(APPS_OBJS) $(OBJS) $(LIBS)

clean:
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm  -f $(ROOT_DIR)/$(UAL_ARCH)/obj/* $(ROOT_DIR)/$(UAL_ARCH)/tracker

$(ROOT_DIR)/$(UAL_ARCH)/obj/%.o : $(SRC_DIR)/%.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;


