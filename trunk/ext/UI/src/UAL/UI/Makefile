include $(UAL)/ext/UI/src/Makefile.config

include ./Objects

compile : $(OBJS) 
	if [ ! -d $(UAL)/ext/UI/include/UAL/UI ] ; then mkdir $(UAL)/ext/UI/include/UAL/UI ; fi;
	cp Argument.hh Arguments.hh Shell.hh OpticsCalculator.hh BunchGenerator.hh $(UAL)/ext/UI/include/UAL/UI/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL)/ext/UI/include/UAL/UI

