include $(UAL)/env/$(UAL_ARCH)/Makefile.config

include ./Objects

compile : $(OBJS)
	(if [ !  -d $(UAL_TIBETAN)/include/TIBETAN/Base ]; \
		then mkdir $(UAL_TIBETAN)/include/TIBETAN/Base; fi;)
	cp *.hh $(UAL_TIBETAN)/include/TIBETAN/Base/.	

clean:
	rm -f $(OBJS)
	rm -rf $(UAL_TIBETAN)/include/TIBETAN/Base
