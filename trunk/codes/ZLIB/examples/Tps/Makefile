include $(UAL)/env/$(UAL_ARCH)/Makefile.config

INC      = -I$(UAL_ZLIB)/src # Included files 
LIBS     = -L$(UAL_ZLIB)/lib/$(UAL_ARCH) # Libraries
LIBS 	+= -lZTps -lm

compile : ./$(UAL_ARCH)/tps ./$(UAL_ARCH)/vtps

clean :
	rm -rf ./$(UAL_ARCH)


$(UAL_ARCH)/tps  :  ./src/tps.cc
	$(LD) -o $@  $(LDFLAGS) $(INC) ./src/tps.cc  $(LIBS)

$(UAL_ARCH)/vtps :  ./src/vtps.cc
	$(LD) -o $@  $(LDFLAGS) $(INC) ./src/vtps.cc $(LIBS)





