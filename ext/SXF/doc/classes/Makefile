include $(UAL)/env/$(UAL_ARCH)/Makefile.config

CXX2HTML = $(SXF)/tools/cxx2html/bin/cxx2html.pl

compile : 
	rm -rf ual_sxf/*.html ual_sxf/buckets/*.html 
	$(PERL) $(CXX2HTML)  -root $(UAL_SXF)/src -- \
	$(UAL_SXF)/src/ual_sxf/*.hh $(UAL_SXF)/src/ual_sxf/buckets/*.hh

clean:
	rm -rf ual_sxf/*.html ual_sxf/buckets/*.html  