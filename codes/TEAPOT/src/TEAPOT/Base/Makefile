include $(UAL_TEAPOT)/src/Makefile.config

include ./Objects

compile : $(OBJS)
	(if [ !  -d $(UAL_TEAPOT)/include/TEAPOT/Base ]; \
		then mkdir $(UAL_TEAPOT)/include/TEAPOT/Base; fi;)
	cp *.hh $(UAL_TEAPOT)/include/TEAPOT/Base/.	

clean:
	rm -f $(OBJS)
	rm -rf $(UAL_TEAPOT)/include/TEAPOT/Base
