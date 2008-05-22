include $(UAL_PAC)/src/Makefile.config

LIBS 	+= -lPac -L$(UAL_ZLIB)/lib/$(UAL_ARCH)

LDIR = $(UAL_PAC)/lib/$(UAL_ARCH)

OBJS = 	$(LDIR)/obj/PacSurveyData.o \
        $(LDIR)/obj/PacSurvey.o \
	$(LDIR)/obj/PacSurveyMap.o \
	$(LDIR)/obj/PacSurveyDrift.o \
	$(LDIR)/obj/PacSurveySbend.o

compile : $(LDIR)/libPacSurvey.so  $(LDIR)/obj/dummy

clean:
	rm -f $(LDIR)/libPacSurvey.so	
	rm -f $(OBJS)
	rm -f $(LDIR)/obj/dummy.o $(LDIR)/obj/dummy


$(LDIR)/libPacSurvey.so :  $(OBJS)
	$(DLD) $(DLDFLAGS) -o $@ $(OBJS) $(LIBS)

# Force template instantiation
$(LDIR)/obj/dummy : $(LDIR)/obj/dummy.o $(OBJS)
	$(LD) $(LDFLAGS) $(INC) -o $@ $(LDIR)/obj/dummy.o $(OBJS) $(LIBS)
	cp /dev/null $(LDIR)/obj/dummy
