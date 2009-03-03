DIRS = ./UAL ./ZLIB ./PAC ./TEAPOT ./ACCSIM ./SPINK ./TIBETAN ./SIMBAD # ./ICE ./AIM 

#ifdef BETACOOL
#DIRS += ./BETACOOL
#endif

compile:  
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done

clean: 
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done
