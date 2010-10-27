include $(UAL_ETEAPOT)/src/Makefile.config

include ./Objects

compile : $(OBJS)
	(if [ !  -d $(UAL_ETEAPOT)/include/ETEAPOT/Base ]; \
		then mkdir $(UAL_ETEAPOT)/include/ETEAPOT/Base; fi;)
	cp *.hh $(UAL_ETEAPOT)/include/ETEAPOT/Base/.

clean:
	rm -f $(OBJS)
	rm -rf $(UAL_ETEAPOT)/include/ETEAPOT/Base
