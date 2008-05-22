include $(SXF)/config/$(SXF_ARCH)/Makefile.config

LDIR = $(SXF)/lib/$(SXF_ARCH)
YLIBS = -ll -ly -lm

OBJS	= \
	$(LDIR)/obj/OStream.o             \
	$(LDIR)/obj/ElemBucket.o          \
	$(LDIR)/obj/ElemBucketRegistry.o  \
	$(LDIR)/obj/AcceleratorNode.o     \
	$(LDIR)/obj/Element.o             \
	$(LDIR)/obj/Sequence.o            \
	$(LDIR)/obj/SequenceStack.o       \
	$(LDIR)/obj/NodeRegistry.o        \
	$(LDIR)/obj/AcceleratorReader.o   \
	$(LDIR)/obj/y.tab.o               \
	$(LDIR)/obj/lex.yy.o 	 

HASHES = \
	$(LDIR)/obj/hashes/ElemAlignHash.o       \
	$(LDIR)/obj/hashes/ElemApertureHash.o    \
	$(LDIR)/obj/hashes/ElemBeamBeamHash.o    \
	$(LDIR)/obj/hashes/ElemBendHash.o        \
	$(LDIR)/obj/hashes/ElemElSeparatorHash.o \
	$(LDIR)/obj/hashes/ElemMultipoleHash.o   \
	$(LDIR)/obj/hashes/ElemRfCavityHash.o    \
	$(LDIR)/obj/hashes/ElemSolenoidHash.o    \
	$(LDIR)/obj/hashes/ElemEmptyHash.o       \
	$(LDIR)/obj/hashes/ElemCollimatorHash.o 
   
compile : $(LDIR)/libSXF.$(SOEXT)

clean:
	rm -f $(LDIR)/libSXF.$(SOEXT)
	rm -f $(LDIR)/obj/*.o $(LDIR)/obj/hashes/*.o
	rm -f y.tab.h y.tab.cc lex.yy.cpp Perfect_Hash.hh

Perfect_Hash.hh : elements.gperf
	$(GPERF) -a -p  -j1 -g -o -t -T -k1,2 -L C++ -Z SXF_Perfect_Hash -N sxf_elements_gperf <  elements.gperf  > Perfect_Hash.hh

y.tab.h y.tab.cc : sxf.yy
	@-$(RM) -f $@
	$(YACC) $(YFLAGS) sxf.yy
	if [ -f sxf.tab.hh ]; then mv -f sxf.tab.hh y.tab.h; fi;	
	if [ -f sxf.tab.cc ]; then mv -f sxf.tab.cc y.tab.cc; fi;
	if [ -f sxf.yy.tab.h ]; then mv -f sxf.yy.tab.h y.tab.h; fi;	
	if [ -f sxf.yy.tab.c ]; then mv -f sxf.yy.tab.c y.tab.cc; fi;

lex.yy.cpp: sxf.ll y.tab.h
	$(LEX) $(LFLAGS) sxf.ll > $@
	@sed -e "s/unistd/stdio/g" -e "s@#include <@#include /**/ <@"  < $@ >> /tmp/$@
	@cp /tmp/$@ $@
	@-$(RM) -f /tmp/$@

$(LDIR)/libSXF.$(SOEXT) :  Perfect_Hash.hh $(OBJS) $(HASHES)
	$(DLD) $(DLDFLAGS) -o $@ $(OBJS)  $(HASHES) $(LIBS)
#	rm -f y.tab.h y.tab.cc lex.yy.cpp 

$(LDIR)/obj/lex.yy.o: lex.yy.cpp
	$(CC) $(CCFLAGS) $(INC) $(LIBS) -DYY_NEVER_INTERACTIVE -c $< -o $@;
