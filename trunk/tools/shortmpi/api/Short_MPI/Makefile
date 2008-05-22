
compile:
	if [ ! -d ./$(UAL_ARCH) ] ; then (mkdir ./$(UAL_ARCH); ); fi;
	swig -perl5 -c++ -I$(MPIHOME)/include \
		-o $(UAL_MPI_PERL)/api/Short_MPI/$(UAL_ARCH)/Short_MPI_wrap.c \
		$(UAL_MPI_PERL)/src/MPI/Short_MPI.i
	cp Makefile.PL ./$(UAL_ARCH)/.
	(cd $(UAL_ARCH); perl Makefile.PL; make;)
clean :
	rm -rf ./$(UAL_ARCH);
	rm -rf ../blib/$(UAL_ARCH);
