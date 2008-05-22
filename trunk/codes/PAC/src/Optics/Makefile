include $(UAL_PAC)/src/Makefile.config

LDIR = $(UAL_PAC)/lib/$(UAL_ARCH)

OBJS  = $(LDIR)/obj/PacVTps.o \
	$(LDIR)/obj/PacTMap.o \
	$(LDIR)/obj/PacTwissData.o \
	$(LDIR)/obj/PacChromData.o 
#	$(LDIR)/obj/LinearMapper.o 	


INC   += -I$(UAL_ZLIB)/src
INC   += -I$(UAL_CORE)/src

LIBS  += -L$(UAL_PAC)/lib/$(UAL_ARCH)  -lPac 
LIBS  += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) -lZTps
LIBS  += -L$(UAL_CORE)/lib/$(UAL_ARCH) -lUal

compile : $(LDIR)/libPacOptics.so $(LDIR)/obj/dummy

clean:
	rm -f $(LDIR)/libPacOptics.so
	rm -f $(LDIR)/obj/*.o $(LIDIR)/obj/*.rpo $(LDIR)/obj/dummy

$(LDIR)/libPacOptics.so :  $(OBJS)
	$(DLD) $(DLDFLAGS) -o $@ $(OBJS) $(LIBS)

# Force template instantiation
$(LDIR)/obj/dummy : $(LDIR)/obj/dummy.o $(OBJS)
	$(LD) $(LDFLAGS) $(INC) -o $@  $(LDIR)/obj/dummy.o $(OBJS) $(LIBS)
	cp /dev/null $(LDIR)/obj/dummy
