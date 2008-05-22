include $(UAL_ZLIB)/config/$(UAL_ARCH)/Makefile.config

LIBS += -lZTps -lm

compile : tps vtps

clean:
	rm -f ./*.o ./out/*.new tps vtps 

tps  :  ../src/tps.cc
	$(LD) -o $@  $(LDFLAGS) $(INC) ../src/tps.cc  $(LIBS)

vtps :  ../src/vtps.cc
	$(LD) -o $@  $(LDFLAGS) $(INC) ../src/vtps.cc $(LIBS)
