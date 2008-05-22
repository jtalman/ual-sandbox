include $(UAL)/env/$(UAL_ARCH)/Makefile.config

INC += -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux -I$(UAL_CORE)/src
LIBS += -L$(JAVA_HOME)/jre/lib/i386 -L$(JAVA_HOME)/jre/lib/i386/client -ljava  -ljvm -lverify

include ./Objects

compile : $(OBJS) 
	if [ ! -d $(UAL_CORE)/include/UAL/JVM ] ; then mkdir $(UAL_CORE)/include/UAL/JVM ; fi;
	cp *.hh $(UAL_CORE)/include/UAL/JVM/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_CORE)/include/UAL/JVM

$(UAL_CORE)/lib/$(UAL_ARCH)/obj/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;



