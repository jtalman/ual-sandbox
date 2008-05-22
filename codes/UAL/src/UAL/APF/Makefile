include $(UAL)/env/$(UAL_ARCH)/Makefile.config

INC      = -I$(UAL_CORE)/src       
INC     += -I$(UAL)/tools/include 

LIBS     = -L$(UAL_CORE)/lib/$(UAL_ARCH) 
LIBS    += -L$(UAL)/tools/$(UAL_ARCH) -lpcre 

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_CORE)/include/UAL/APF ] ; then mkdir $(UAL_CORE)/include/UAL/APF ; fi;
	cp *.hh $(UAL_CORE)/include/UAL/APF/.

clean:
	rm -f $(OBJS) 
	rm -rf $(UAL_CORE)/include/UAL/APF

$(UAL_CORE)/lib/$(UAL_ARCH)/obj/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;
