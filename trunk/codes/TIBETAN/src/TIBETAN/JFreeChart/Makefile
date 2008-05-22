include $(UAL)/env/$(UAL_ARCH)/Makefile.config

INC += -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux -I$(UAL_CORE)/src -I$(UAL_TIBETAN)/src
LIBS += -L$(JAVA_HOME)/jre/lib/i386 -L$(JAVA_HOME)/jre/lib/i386/client -ljava  -ljvm -lverify

include ./Objects

compile : $(OBJS) 
	if [ ! -d $(UAL_TIBETAN)/include/TIBETAN/JFreeChart ] ; \
		then mkdir $(UAL_TIBETAN)/include/TIBETAN/JFreeChart ; fi;
	cp *.hh $(UAL_TIBETAN)/include/TIBETAN/JFreeChart/.

clean:	
	rm -f $(OBJS) 
	rm -rf $(UAL_TIBETAN)/include/TIBETAN/JFreeChart

$(UAL_TIBETAN)/lib/$(UAL_ARCH)/obj/%.o : %.cc
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@;



