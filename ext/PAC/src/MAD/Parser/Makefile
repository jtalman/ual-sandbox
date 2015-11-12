include $(UAL)/env/$(UAL_ARCH)/Makefile.config

CORE = $(UAL_EXTRA)/PAC

compile : $(CORE)/bin/$(UAL_ARCH)/mad2smf

clean :
	rm -f $(CORE)/bin/$(UAL_ARCH)/mad2smf

$(CORE)/bin/$(UAL_ARCH)/mad2smf : parser.l parser.y
	$(LEX)  parser.l
	$(YACC) parser.y
	$(cc) parser.tab.c $(YACC_LIB)
	mv a.out $(CORE)/bin/$(UAL_ARCH)/mad2smf;
	rm  parser.tab.c
	rm  lex.yy.c
