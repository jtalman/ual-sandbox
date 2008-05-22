
DIRS = 	./src ./doc

LIBDIRS= $(SXF)/api/ROOT \
	$(UAL_CORE)/api/ROOT \
	$(UAL_ZLIB)/api/ROOT \
	$(UAL_PAC)/api/ROOT  \
	$(UAL_AIM)/api/ROOT  

compile:  
	if [ ! -d ./lib ] ; then (mkdir ./lib; ); fi;
	if [ ! -d ./lib/$(UAL_ARCH) ] ; \
		then (mkdir ./lib/$(UAL_ARCH); mkdir ./lib/$(UAL_ARCH)/obj; ); fi;
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean: 
	@for dir in $(DIRS) ; do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done

cleanUALROOT:
	override DIRS += $(LIBDIRS)
	@for dir in $(DIRS) $(LIBDIRS)  ; do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done

compileUALROOT:
	if [ ! -d ./lib ] ; then (mkdir ./lib; ); fi;
	if [ ! -d ./lib/$(UAL_ARCH) ] ; \
		then (mkdir ./lib/$(UAL_ARCH); mkdir ./lib/$(UAL_ARCH)/obj; ); fi;
	@for dir in $(LIBDIRS) $(DIRS) ; do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
