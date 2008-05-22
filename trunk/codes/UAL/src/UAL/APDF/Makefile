include $(UAL)/env/$(UAL_ARCH)/Makefile.config

INC     += -I$(UAL_CORE)/src   
INC     += -I$(UAL)/tools/include -I$(UAL_LIBXM2)  

LIBS     = -L$(UAL_CORE)/lib/$(UAL_ARCH)
LIBS    += -L$(UAL)/tools/$(UAL_ARCH) -lpcre -lxml2           


include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_CORE)/include/UAL/APDF ] ; then mkdir $(UAL_CORE)/include/UAL/APDF ; fi;
	cp APDF_Builder.hh $(UAL_CORE)/include/UAL/APDF/.

clean:
	rm -f $(OBJS) 
	rm -rf $(UAL_CORE)/include/UAL/APDF

$(UAL_CORE)/lib/$(UAL_ARCH)/obj/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;

