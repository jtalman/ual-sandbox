include $(UAL_AIM)/src/Makefile.config

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_AIM)/include/AIM/BTF ] ; \
		then mkdir $(UAL_AIM)/include/AIM/BTF ; fi;
	cp *.hh $(UAL_AIM)/include/AIM/BTF/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_AIM)/include/AIM/BTF
