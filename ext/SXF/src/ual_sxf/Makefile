include $(UAL_SXF)/src/Makefile.config

LDIR = $(UAL_SXF)/lib/$(UAL_ARCH)
# LIBS +=  -lPacSMF -lPacCommon -lPacOptics  -lZTps -lSXF

BACKEND = \
	$(LDIR)/obj/ElemBucket.o             \
	$(LDIR)/obj/ElemBucketRegistry.o     \
	$(LDIR)/obj/Element.o                \
	$(LDIR)/obj/Error.o                  \
	$(LDIR)/obj/NodeRegistry.o           \
	$(LDIR)/obj/Sequence.o               \
	$(LDIR)/obj/AcceleratorReader.o      \
	$(LDIR)/obj/Parser.o        


BUCKETS	= \
	$(LDIR)/obj/buckets/ElemError.o      \
	$(LDIR)/obj/buckets/ElemMultipole.o  \
	$(LDIR)/obj/buckets/ElemMltEntry.o   \
	$(LDIR)/obj/buckets/ElemMltBody.o    \
	$(LDIR)/obj/buckets/ElemMltExit.o    \
	$(LDIR)/obj/buckets/ElemSolenoid.o   \
	$(LDIR)/obj/buckets/ElemEmpty.o      \
	$(LDIR)/obj/buckets/ElemBend.o       \
	$(LDIR)/obj/buckets/ElemRfCavity.o   \
	$(LDIR)/obj/buckets/ElemAlign.o      \
	$(LDIR)/obj/buckets/ElemKicker.o     \
	$(LDIR)/obj/buckets/ElemCollimator.o \
	$(LDIR)/obj/buckets/ElemAperture.o

OBJS = $(BUCKETS) $(BACKEND)

compile : Perfect_Hash.hh $(LDIR)/obj/dummy $(LDIR)/libUalSXF.so

clean:
	rm -f $(LDIR)/libUalSXF.so
	rm -f $(LDIR)/obj/*.o $(LDIR)/obj/buckets/*.o $(LDIR)/obj/elements/*.o
	rm -f Perfect_Hash.hh

Perfect_Hash.hh : elements.gperf
	$(GPERF) -a -p  -j1 -g -o -t -T -k1,2  -L C++ -Z UAL_SXF_Perfect_Hash -N smf_elements_gperf \
	<  elements.gperf  > Perfect_Hash.hh

$(LDIR)/libUalSXF.so :  $(OBJS)
	$(DLD) $(DLDFLAGS) -o $@ $(OBJS) $(LIBS) \
		-lPacSMF -lPacOptics  -lPac -lZTps -lSXF

# Force template instantiation
$(LDIR)/obj/dummy : $(LDIR)/obj/dummy.o $(OBJS)
	$(LD) $(LDFLAGS) $(INC) -o $@  $(LDIR)/obj/dummy.o $(OBJS) $(LIBS) \
		-lPacSMF -lPacOptics  -lPac -lZTps -lSXF
	cp /dev/null $(LDIR)/obj/dummy

