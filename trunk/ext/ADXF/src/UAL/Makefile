include $(UAL)/ext/ADXF/src/Makefile.config

DIRS = ./ADXF

LDIR = $(UAL)/ext/ADXF/lib/$(UAL_ARCH)

OBJS = 
include ./ADXF/Objects

compile : $(LDIR)/obj/dummy.o
	(if [ !  -d $(UAL)/ext/ADXF/include/UAL ]; then mkdir  $(UAL)/ext/ADXF/include/UAL; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(LD) $(LDFLAGS) $(INC) -o $(LDIR)/obj/dummy  $(LDIR)/obj/dummy.o $(OBJS) $(LIBS)
	cp /dev/null $(LDIR)/obj/dummy
	$(DLD) $(DLDFLAGS) -o $(LDIR)/libUalADXF2.so $(OBJS) $(LIBS)
	rm -r $(LDIR)/obj/*.o $(LDIR)/obj/elements/*.o $(LDIR)/obj/handlers/*.o

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL)/ext/ADXF/include/UAL
	rm -rf $(LDIR)
