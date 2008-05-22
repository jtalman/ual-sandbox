include $(UAL_AIM)/src/Makefile.config

include ./Objects

compile : $(OBJS)
	if [ ! -d $(UAL_AIM)/include/AIM/BPM ] ; \
		then mkdir $(UAL_AIM)/include/AIM/BPM ; fi;
	cp *.hh $(UAL_AIM)/include/AIM/BPM/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_AIM)/include/AIM/BPM
