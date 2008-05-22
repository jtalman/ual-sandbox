include $(UAL)/env/$(UAL_ARCH)/Makefile.config

INC      = -I$(UAL_PAC)/src -I$(UAL_CORE)/include 
LIBS     = -L$(UAL_PAC)/lib/$(UAL_ARCH) -L$(UAL_CORE)/lib/$(UAL_ARCH) -lUal -ldl 

LIBS 	+= -lPac -lPacOptics 
LIBS 	+= -L$(UAL_ZLIB)/lib/$(UAL_ARCH) -lZTps -lm 

compile: beam

clean:
	rm -rf ./beam
	rm -f $(UAL_PAC)/lib/$(UAL_ARCH)/obj/beam.o 


beam : $(UAL_PAC)/lib/$(UAL_ARCH)/obj/beam.o
	$(LD) -o $@  $(LDFLAGS) $(INC) $(UAL_PAC)/lib/$(UAL_ARCH)/obj/beam.o $(LIBS)


$(UAL_PAC)/lib/$(UAL_ARCH)/obj/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;
