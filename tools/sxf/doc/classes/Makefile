include $(SXF)/config/$(SXF_ARCH)/Makefile.config

compile : 
	$(RM) -rf sxf/*.html sxf/hashes/*.html echo_sxf/*.html 
	$(PERL) $(CXX2HTML)  -root $(CORE)/src -- \
	$(CORE)/src/sxf/*.hh $(CORE)/src/sxf/hashes/*.hh      \
	$(CORE)/src/echo_sxf/*.hh
clean:
	$(RM) -rf sxf/*.html sxf/hashes/*.html echo_sxf/*.html 