%module Short_MPI
%include "typemaps.i"
%{
extern "C" {
#include "mpi.h"
}
%}

// Tells SWIG to treat char*** as a special case
%typemap(perl5,in) char *** {
	AV *tempav;
	I32 len;
	int i;
	char** argv;
	SV **tv;
	if(!SvROK($input)) croak("$input is not a reference.");
	if(SvTYPE(SvRV($input)) != SVt_PVAV) croak("$input is not an array.");
	tempav = (AV*) SvRV($input);
	len = av_len(tempav);
	argv = new char*[len + 2];
	for(i = 0; i <= len; i++){
		tv = av_fetch(tempav, i, 0);
		argv[i] = (char *) SvPV(*tv, PL_na);
        }
	argv[i] = 0;
	$1 = &argv;
}

%typemap(perl5,in) int *result (int ivalue) {
	SV* tempsv;
	if(!SvROK($input)) croak("expected a reference\n");

	tempsv = SvRV($input);
	$1 = &ivalue;
}

%typemap(perl5,argout) int *result {
	SV* tempsv;
	tempsv = SvRV($arg);
	sv_setiv(tempsv, (int) *($1));
}

%constant int MPI_COMM_WORLD = MPI_COMM_WORLD;

typedef int MPI_Comm;

extern "C" int MPI_Init(int *INPUT, char ***argv);

extern "C" int MPI_Finalize();

extern "C" int MPI_Initialized(int *result);

extern "C" int MPI_Comm_size(MPI_Comm comm, int *result);

extern "C" int MPI_Comm_rank(MPI_Comm comm, int *result);

extern "C" int MPI_Barrier(MPI_Comm comm);

extern "C" double MPI_Wtime();










