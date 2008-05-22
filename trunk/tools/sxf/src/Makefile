include $(SXF)/config/$(SXF_ARCH)/Makefile.config

DIRS = ./sxf ./echo_sxf

compile :
	(if [ !  -d ../lib ]; then mkdir  ../lib; fi;)
	(if [ !  -d ../lib/$(SXF_ARCH) ]; then mkdir  ../lib/$(SXF_ARCH); fi;)
	(if [ !  -d ../lib/$(SXF_ARCH)/obj ]; then mkdir  ../lib/$(SXF_ARCH)/obj; fi; )
	(if [ !  -d ../lib/$(SXF_ARCH)/obj/hashes ]; then mkdir  ../lib/$(SXF_ARCH)/obj/hashes; fi; )		
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf ../lib/$(SXF_ARCH)

