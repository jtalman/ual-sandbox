include ../../Makefile.config

LDIR = $(UAL_SXF)/lib/$(UAL_ARCH)

HHS = 

include ./Objects

compile : Perfect_Hash.hh $(OBJS)
	(if [ !  -d $(UAL_SXF)/include/UAL/SXF ]; \
		then mkdir $(UAL_SXF)/include/UAL/SXF; fi;)
	(if [ !  -d $(UAL_SXF)/include/UAL/SXF ]; then mkdir  $(UAL_SXF)/include/UAL/SXF; fi;)
	cp *.hh ./buckets/*.hh $(UAL_SXF)/include/UAL/SXF/.	

clean:
	rm -rf $(UAL_SXF)/include/UAL/SXF
	rm -f $(OBJS)
	rm -f Perfect_Hash.hh

Perfect_Hash.hh : elements.gperf
	$(GPERF) -a -p  -j1 -g -o -t -T -k1,2  -L C++ -Z UAL_SXF_Perfect_Hash -N smf_elements_gperf \
	<  elements.gperf  > Perfect_Hash.hh

