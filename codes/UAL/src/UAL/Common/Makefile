include $(UAL)/env/$(UAL_ARCH)/Makefile.config

INC      = -I$(UAL_CORE)/src                # Included files 
LIBS     = -L$(UAL_CORE)/lib/$(UAL_ARCH)    # Libraries

include ./Objects

compile : $(OBJS) 
	if [ ! -d $(UAL_CORE)/include/UAL/Common ] ; then mkdir $(UAL_CORE)/include/UAL/Common ; fi;
	cp *.hh $(UAL_CORE)/include/UAL/Common/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_CORE)/include/UAL/Common

$(UAL_CORE)/lib/$(UAL_ARCH)/obj/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;


