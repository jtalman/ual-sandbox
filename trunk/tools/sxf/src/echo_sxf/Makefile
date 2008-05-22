include $(SXF)/config/$(SXF_ARCH)/Makefile.config

LDIR = $(SXF)/lib/$(SXF_ARCH)

# LIBS += -lSXF

OBJS	= \
	$(LDIR)/obj/EchoElemBucket.o             \
	$(LDIR)/obj/EchoElemError.o              \
	$(LDIR)/obj/EchoElemBucketRegistry.o     \
	$(LDIR)/obj/EchoElement.o                \
	$(LDIR)/obj/EchoError.o                  \
	$(LDIR)/obj/EchoSequence.o               \
	$(LDIR)/obj/EchoNodeRegistry.o           \
	$(LDIR)/obj/EchoAcceleratorReader.o      \
	$(LDIR)/obj/EchoParser.o        

compile : $(LDIR)/libEchoSXF.$(SOEXT)

clean :
	rm -f $(LDIR)/libEchoSXF.$(SOEXT)	
	rm -f $(LDIR)/obj/*.o

$(LDIR)/libEchoSXF.$(SOEXT) :  $(OBJS)
	$(DLD) $(DLDFLAGS) -o $@ $(OBJS) -lSXF $(LIBS)