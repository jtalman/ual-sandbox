include $(UAL)/env/$(UAL_ARCH)/Makefile.config

INC      = -I$(UAL_PAC)/src -I$(UAL_CORE)/include 
LIBS     = -L$(UAL_PAC)/lib/$(UAL_ARCH) -L$(UAL_CORE)/lib/$(UAL_ARCH) -lUal -ldl 

INC  	+= -I/$(UAL_ZLIB)/src
LIBS 	+= -lPacOptics -lPac -L$(UAL_ZLIB)/lib/$(UAL_ARCH) -lZTps -lm 

compile : optics

clean:
	rm -rf ./optics
	rm -f $(UAL_PAC)/lib/$(UAL_ARCH)/obj/optics.o
	rm -rf ./out/*.new

optics : $(UAL_PAC)/lib/$(UAL_ARCH)/obj/optics.o
	$(LD) -o $@  $(LDFLAGS) $(INC) $(UAL_PAC)/lib/$(UAL_ARCH)/obj/optics.o $(LIBS)

$(UAL_PAC)/lib/$(UAL_ARCH)/obj/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;



