include $(UAL)/env/$(UAL_ARCH)/Makefile.config

INC      = -I$(UAL_ZLIB)/src # Included files 
LIBS     = -L$(UAL_ZLIB)/lib/$(UAL_ARCH) # Libraries

LDIR = $(UAL_ZLIB)/lib/$(UAL_ARCH)

OBJS  = $(LDIR)/obj/GlobalTable.o \
	$(LDIR)/obj/Space.o \
	$(LDIR)/obj/Vector.o \
	$(LDIR)/obj/Tps.o \
	$(LDIR)/obj/VTps.o

compile : $(LDIR)/libZTps.so
	if [ ! -d $(UAL_ZLIB)/include/ZLIB/Tps ] ; then mkdir $(UAL_ZLIB)/include/ZLIB/Tps ; fi;
	cp *.hh $(UAL_ZLIB)/include/ZLIB/Tps/.

clean:
	rm -f $(LDIR)/libZTps.so
	rm -f $(LDIR)/obj/*.o
	rm -rf $(UAL_ZLIB)/include/ZLIB/Tps

$(LDIR)/libZTps.so :  $(OBJS)
	$(DLD) $(DLDFLAGS) -o $@ $(OBJS) $(LIBS)

$(UAL_ZLIB)/lib/$(UAL_ARCH)/obj/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;



