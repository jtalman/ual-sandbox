include ../../Makefile.config

LDIR = $(UAL)/ext/ADXF/lib/$(UAL_ARCH)

HHS = 

include ./Objects

compile : $(OBJS)
	(if [ !  -d $(UAL)/ext/ADXF/include/UAL/ADXF ]; \
		then mkdir $(UAL)/ext/ADXF/include/UAL/ADXF; fi;)
	(if [ !  -d $(UAL)/ext/ADXF/include/UAL/ADXF ]; then mkdir  $(UAL)/ext/ADXF/include/UAL/ADXF; fi;)
	cp *.hh ./elements/*.hh $(UAL)/ext/ADXF/include/UAL/ADXF/.	

clean:
	rm -rf $(UAL)/ext/ADXF/include/UAL/ADXF
	rm -f $(OBJS)

