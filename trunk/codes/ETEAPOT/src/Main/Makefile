include $(UAL_ETEAPOT)/src/Makefile.config

LDIR = $(UAL_ETEAPOT)/lib/$(UAL_ARCH)

LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacSMF
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacOptics 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPac 
LIBS += -L$(UAL_PAC)/lib/$(UAL_ARCH) -lPacSurvey
LIBS += -L$(UAL_ZLIB)/lib/$(UAL_ARCH) -lZTps

include ./Objects

compile : $(OBJS)

clean:
	rm -f $(OBJS)

# Force template instantiation
$(LDIR)/obj/dummy : $(LDIR)/obj/dummy.o $(OBJS)
	$(LD) $(LDFLAGS) $(INC) -o $@  $(LDIR)/obj/dummy.o $(OBJS) $(LIBS)
	cp /dev/null $(LDIR)/obj/dummy
