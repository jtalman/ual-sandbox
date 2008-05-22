include $(UAL)/env/$(UAL_ARCH)/Makefile.config

PLS =  	$(UAL_ZLIB)/doc/examples/tps.txt \
	$(UAL_ZLIB)/doc/examples/vtps.txt

DIRS = ./src

compile : 
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf 	$(PLS)

$(UAL_ZLIB)/doc/examples/%.txt : $(CORE)/examples/Tps/api/%.pl
	if [ ! -d ./examples ] ; then mkdir ./examples ; fi;
	cp $< $@
