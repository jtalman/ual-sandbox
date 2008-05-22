include $(UAL)/env/$(UAL_ARCH)/Makefile.config

DIRS = ./Tps

compile :
	@for dir in $(DIRS); do \
		(cd $$dir; \
			if [ ! -d ./$(UAL_ARCH) ] ; \
				then (mkdir ./$(UAL_ARCH); mkdir ./$(UAL_ARCH)/out; \
				cp ./src/Makefile ./$(UAL_ARCH)/. ;); \
			fi; \
			if [ -f ./Makefile ]; \
				then $(MAKE) compile; \
			fi; \
		); \
	done
clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done

