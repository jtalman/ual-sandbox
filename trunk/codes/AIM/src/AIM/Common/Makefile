include $(UAL_AIM)/src/Makefile.config

include ./Objects

compile : $(OBJS)
	(if [ !  -d $(UAL_AIM)/include/AIM/Common ]; \
		then mkdir $(UAL_AIM)/include/AIM/Common; fi;)
	cp *.hh $(UAL_AIM)/include/AIM/Common/.	

clean:
	rm -f $(OBJS)
	rm -rf $(UAL_AIM)/include/AIM/Common
