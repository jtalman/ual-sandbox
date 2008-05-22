include $(UAL_SXF)/src/Makefile.config

DIRS = ./SXF

LDIR = $(UAL_SXF)/lib/$(UAL_ARCH)

OBJS = 
# include ./SMF/Objects
include ./SXF/Objects

compile : $(LDIR)/obj/dummy.o
	(if [ !  -d $(UAL_SXF)/include/UAL ]; then mkdir  $(UAL_SXF)/include/UAL; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(LD) $(LDFLAGS) $(INC) -o $(LDIR)/obj/dummy  $(LDIR)/obj/dummy.o $(OBJS) $(LIBS)
	cp /dev/null $(LDIR)/obj/dummy
	$(DLD) $(DLDFLAGS) -o $(LDIR)/libUalSXF2.so $(OBJS) $(LIBS)
	rm -r $(LDIR)/obj/*.o $(LDIR)/obj/buckets/*.o

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_SXF)/include/UAL
	rm -rf $(LDIR)
