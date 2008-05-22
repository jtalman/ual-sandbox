include $(SXF)/config/$(SXF_ARCH)/Makefile.config

compile : ./$(SXF_ARCH)/sxf_tester

clean:
	rm -f ./$(SXF_ARCH)/obj/*.o  ./out/*.new ./$(SXF_ARCH)/sxf_tester 

$(SXF_ARCH)/sxf_tester  :  ./$(SXF_ARCH)/obj/sxf_tester.o
	$(LD) -o $@  $(INC) ./$(SXF_ARCH)/obj/sxf_tester.o -lEchoSXF -lSXF $(LIBS)

$(SXF_ARCH)/obj/sxf_tester.o  :  ./src/sxf_tester.cc
	$(CC) $(CCFLAGS) $(INC) $(LIBS) -c $< -o $@;
