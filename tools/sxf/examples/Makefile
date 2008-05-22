include $(SXF)/config/$(SXF_ARCH)/Makefile.config

DIRS = ./echo

compile :
	@for dir in $(DIRS); do \
		(cd $$dir; mkdir ./$(SXF_ARCH);  mkdir ./$(SXF_ARCH)/obj; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; rm -rf ./$(SXF_ARCH); fi;); \
	done 
