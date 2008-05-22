include $(UAL_PAC)/src/Makefile.config

INC  	+=  -I$(UAL_CORE)/include 
INC  	+=  -I$(UAL_ZLIB)/src
INC     +=  -I$(UAL)/tools/include

LIBS 	+=  -lPacOptics -lPac -L$(UAL_ZLIB)/lib/$(UAL_ARCH) -lZTps
LIBS 	+=  -L$(UAL_CORE)/lib/$(UAL_ARCH) -lUal
LIBS    +=  -L$(UAL)/tools/lib/$(UAL_ARCH)  -lpcre -lpcreposix

LDIR = $(UAL_PAC)/lib/$(UAL_ARCH)

OBJS =  $(LDIR)/obj/PacElemKey.o \
	$(LDIR)/obj/PacElemKeys.o \
	$(LDIR)/obj/PacElemBucketKey.o \
	$(LDIR)/obj/PacElemBucketKeys.o \
	$(LDIR)/obj/PacElemBucket.o \
	$(LDIR)/obj/PacElemAttributes.o \
	$(LDIR)/obj/PacGenElement.o \
	$(LDIR)/obj/PacGenElements.o\
	$(LDIR)/obj/PacLine.o \
	$(LDIR)/obj/PacLines.o \
	$(LDIR)/obj/PacLattElement.o \
	$(LDIR)/obj/PacLattice.o \
	$(LDIR)/obj/PacLattices.o \
	$(LDIR)/obj/PacSmf.o 

compile :  $(UAL_PAC)/lib/$(UAL_ARCH)/libPacSMF.so $(LDIR)/obj/dummy

clean:
	rm -f $(UAL_PAC)/lib/$(UAL_ARCH)/libPacSMF.so		
	rm -rf $(LDIR)/obj/*.o 
	rm -rf $(LDIR)/obj/*.rpo $(LDIR)/obj/dummy

$(UAL_PAC)/lib/$(UAL_ARCH)/libPacSMF.so :  $(OBJS)
	$(DLD) $(DLDFLAGS) -o $@ $(OBJS) $(LIBS)

# Force template instantiation
$(LDIR)/obj/dummy : $(LDIR)/obj/dummy.o $(OBJS)
	$(LD) $(LDFLAGS) $(INC) -o $@  $(LDIR)/obj/dummy.o $(OBJS) $(LIBS)
	cp /dev/null $(LDIR)/obj/dummy

