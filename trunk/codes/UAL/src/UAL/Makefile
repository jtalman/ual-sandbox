include $(UAL)/env/$(UAL_ARCH)/Makefile.config

INC      = -I$(UAL_CORE)/src                # Included files 
LIBS     = -L$(UAL_CORE)/lib/$(UAL_ARCH)    # Libraries
LIBS    += -L$(UAL)/tools/lib/$(UAL_ARCH) -lpcre -lxml2

DIRS = ./Common ./SMF ./APF ./APDF

OBJS = 
include ./Common/Objects
include ./SMF/Objects
include ./APF/Objects
include ./APDF/Objects

# ifeq ("$(origin UAL_JAVA)", "environment")
#  DIRS += ./JVM
#  include ./JVM/Objects 
#  LIBS += -L$(JAVA_HOME)/jre/lib/i386 -L$(JAVA_HOME)/jre/lib/i386/client -ljava  -ljvm -lverify
# else
# endif

compile : $(UAL_CORE)/lib/$(UAL_ARCH)/obj/dummy.o
	(if [ !  -d $(UAL_CORE)/include/UAL ]; then mkdir  $(UAL_CORE)/include/UAL; fi;)
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) compile; fi;); \
	done
	$(LD) $(LDFLAGS) $(INC) -o $(UAL_CORE)/lib/$(UAL_ARCH)/obj/dummy   \
		$(UAL_CORE)/lib/$(UAL_ARCH)/obj/dummy.o $(OBJS) $(LIBS)
	cp /dev/null $(UAL_CORE)/lib/$(UAL_ARCH)/obj/dummy
	$(DLD) $(DLDFLAGS) -o $(UAL_CORE)/lib/$(UAL_ARCH)/libUal.so $(OBJS) $(LIBS)

clean :
	@for dir in $(DIRS); do \
		(cd $$dir; if [ -f ./Makefile ]; then $(MAKE) clean; fi;); \
	done 
	rm -rf $(UAL_CORE)/include/UAL
	rm -f $(UAL_CORE)/lib/$(UAL_ARCH)/obj/dummy.o
	rm -f $(UAL_CORE)/lib/$(UAL_ARCH)/libUal.so

$(UAL_CORE)/lib/$(UAL_ARCH)/obj/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;

