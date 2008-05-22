include $(UAL)/env/$(UAL_ARCH)/Makefile.config

INC      = -I$(UAL_SXF)/src
INC     += -I$(UAL_PAC)/src -I$(UAL_ZLIB)/src -I$(UAL_CORE)/include -I$(SXF)/src

LIBS     = -L$(UAL_SXF)/lib/$(UAL_ARCH) 
LIBS    += -L$(UAL_PAC)/lib/$(UAL_ARCH) -L$(UAL_ZLIB)/lib/$(UAL_ARCH) \
	-L$(UAL_CORE)/lib/$(UAL_ARCH) -lUal  -L$(SXF)/lib/$(SXF_ARCH)

compile : ./$(UAL_ARCH)/parser

clean:
	rm -f ./$(UAL_ARCH)/obj/*.o  ./$(UAL_ARCH)/out/*.new ./$(UAL_ARCH)/parser 

$(UAL_ARCH)/parser  :  ./$(UAL_ARCH)/obj/parser.o
	$(CC) -o $@  $(INC) ./$(UAL_ARCH)/obj/parser.o -lUalSXF $(LIBS)	\
		-lPacSMF  -lPacOptics -lPac  -lZTps -lSXF

./$(UAL_ARCH)/obj/parser.o  :  ./src/parser.cc
	if [ ! -d ./$(UAL_ARCH)/obj ] ; then mkdir ./$(UAL_ARCH)/obj ; fi;
	$(CC) $(CCFLAGS) $(INC) $(LIBS) -c $< -o $@;



