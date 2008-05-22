#include <assert.h> 

#ifdef __cplusplus
extern "C" {
#endif
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#ifdef __cplusplus
}
#endif

#ifdef do_open
#undef do_open
#endif

#ifdef do_close
#undef do_close
#endif

#ifdef list
#undef list
#endif

#include "Main/Teapot.h"
#include "Integrator/TeapotIntegrator.h"
#include "Integrator/TeapotDAIntegrator.h"

using namespace PAC;

MODULE = Teapot::Main		PACKAGE = Teapot::Main

Teapot* 
Teapot::new()

void 
Teapot::use(lattice)
	PacLattice* lattice;
	CODE:
	THIS->use(*lattice);

void 
Teapot::DESTROY()

void 
Teapot::makethin()

int 
Teapot::size()

PacLattElement*
Teapot::element(index)
	int index
	CODE:
	char* CLASS = "Pac::LattElement";
	RETVAL = new PacLattElement();
	*RETVAL = THIS->element(index);
	OUTPUT:
	RETVAL	

void 
Teapot::track(...)
	CODE:
	if(items == 3){
            if(SvTYPE(SvRV(ST(1))) == SVt_PVMG)
	    {
                THIS->track(*((Bunch *) SvIV((SV*) SvRV( ST(1) ))), (int ) SvIV(ST(2)));
            } 
            else
	    {
                warn( "Teapot::track(bunch, turns) -- arguments are not a blessed SV reference" );
                XSRETURN_UNDEF;
            }
	}
	if(items == 4){
            if(SvTYPE(SvRV(ST(1))) == SVt_PVMG)
	    {
                THIS->track(*((Bunch *) SvIV((SV*) SvRV( ST(1) ))), (int ) SvIV(ST(2)), (int) SvIV(ST(3)) );
            } 
            else
	    {
                warn( "Teapot::track(bunch, i1, i2) -- arguments are not a blessed SV reference" );
                XSRETURN_UNDEF;
            }
	}

void
Teapot::survey(s, i1, i2)
	PacSurveyData* s
	int i1
	int i2
	CODE:
	THIS->survey(*s, i1, i2);

void
Teapot::clorbit(orbit, beam)
	Position* orbit
	BeamAttributes* beam
	CODE:
	THIS->clorbit(*orbit, *beam);

void
Teapot::trackClorbit(orbit, beam, i1, i2)
	Position* orbit
	BeamAttributes* beam
	int i1
	int i2
	CODE:
	THIS->trackClorbit(*orbit, *beam, i1, i2);

void 
Teapot::steer_(orbit, beam, rads, rdets, method, plane)
	Position* orbit
	BeamAttributes* beam
	SV* rads
	SV* rdets
	int method
	char plane
	CODE:
	AV* ads  = (AV*) SvRV(rads);
	AV* dets = (AV*) SvRV(rdets);
	int yes = 1, i;
	PacVector<int> vads(1 + (int) av_len(ads)), vdets(1 + (int) av_len(dets));
	for(i= 0; i < vads.size(); i++) {
		vads[i] = (int) SvIV( *av_fetch(ads, (I32) i, (I32) yes));
	}
	for(i= 0; i < vdets.size(); i++) {
		vdets[i] = (int) SvIV( *av_fetch(dets, (I32) i, (I32) yes));
	};	
	THIS->steer(*orbit, *beam, vads, vdets, method, plane);

void 
Teapot::ftsteer_(orbit, beam, rhads, rhdets, rvads, rvdets, maxdev, tw, method)
	Position* orbit
	BeamAttributes* beam
	SV* rhads
	SV* rhdets
        SV* rvads
	SV* rvdets
        double maxdev
        PacTwissData* tw
        int method
	CODE:
	AV* hads  = (AV*) SvRV(rhads);
	AV* hdets = (AV*) SvRV(rhdets);
        AV* vads  = (AV*) SvRV(rvads);
	AV* vdets = (AV*) SvRV(rvdets);
	int yes = 1, i;
	PacVector<int> vhads(1 + (int) av_len(hads)), vhdets(1 + (int) av_len(hdets));
        PacVector<int> vvads(1 + (int) av_len(vads)), vvdets(1 + (int) av_len(vdets));
        for(i= 0; i < vhads.size(); i++) {
		vhads[i] = (int) SvIV( *av_fetch(hads, (I32) i, (I32) yes));
               
	}
	for(i= 0; i < vhdets.size(); i++) {
		vhdets[i] = (int) SvIV( *av_fetch(hdets, (I32) i, (I32) yes));
              
	};	
        for(i= 0; i < vvads.size(); i++) {
		vvads[i] = (int) SvIV( *av_fetch(vads, (I32) i, (I32) yes));
              
	}
	for(i= 0; i < vvdets.size(); i++) {
		vvdets[i] = (int) SvIV( *av_fetch(vdets, (I32) i, (I32) yes));
             
	};	
        THIS->ftsteer(*orbit, *beam, vhads, vhdets, vvads, vvdets, maxdev, *tw, method);


void 
Teapot::twissList(tw, beam, orbit)
	PacTwissData* tw
	BeamAttributes* beam
	Position *orbit
	CODE:
        THIS->twissList(*tw, *beam, *orbit);



void 
Teapot::twiss(tw, beam, orbit)
	PacTwissData* tw
	BeamAttributes* beam
	Position *orbit
	CODE:
	THIS->twiss(*tw, *beam, *orbit);


void 
Teapot::trackTwiss(tw, vtps)
	PacTwissData* tw
	PacVTps* vtps
	CODE:
	THIS->trackTwiss(*tw, *vtps);

void 
Teapot::tunethin_(beam, orbit, rb1f, rb1d, mux, muy, method, numtries, tolerance, stepsize)
	BeamAttributes* beam
	Position *orbit
	SV* rb1f
	SV* rb1d
	double mux
	double muy
	char method
	int numtries
	double tolerance
	double stepsize
	CODE:
	AV* b1f = (AV*) SvRV(rb1f);
	AV* b1d = (AV*) SvRV(rb1d);
	int yes = 1, i;
	PacVector<int> vb1f(1 + (int) av_len(b1f)), vb1d(1 + (int) av_len(b1d));
	for(i= 0; i < vb1f.size(); i++) {
		vb1f[i] = (int) SvIV( *av_fetch(b1f, (I32) i, (I32) yes));
	}
	for(i= 0; i < vb1d.size(); i++) {
		vb1d[i] = (int) SvIV( *av_fetch(b1d, (I32) i, (I32) yes));
	};	
	THIS->tunethin(*beam, *orbit, vb1f, vb1d, mux, muy, method, numtries, tolerance, stepsize);

void 
Teapot::eigenTwiss(tw, map)
	PacTwissData* tw
	PacVTps* map
	CODE:
	THIS->eigenTwiss(*tw, *map);

void 
Teapot::trackEigenTwiss(tw, sector)
	PacTwissData* tw
	PacVTps* sector
	CODE:
	THIS->trackEigenTwiss(*tw, *sector);

void 
Teapot::chrom(tw, beam, orbit)
	PacChromData* tw
	BeamAttributes* beam
	Position *orbit
	CODE:
	THIS->chrom(*tw, *beam, *orbit);

void 
Teapot::chromfit_(beam, orbit, rb1f, rb1d, mux, muy, method, numtries, tolerance, stepsize)
	BeamAttributes* beam
	Position *orbit
	SV* rb1f
	SV* rb1d
	double mux
	double muy
	char method
	int numtries
	double tolerance
	double stepsize
	CODE:
	AV* b1f = (AV*) SvRV(rb1f);
	AV* b1d = (AV*) SvRV(rb1d);
	int yes = 1, i;
	PacVector<int> vb1f(1 + (int) av_len(b1f)), vb1d(1 + (int) av_len(b1d));
	for(i= 0; i < vb1f.size(); i++) {
		vb1f[i] = (int) SvIV( *av_fetch(b1f, (I32) i, (I32) yes));
	}
	for(i= 0; i < vb1d.size(); i++) {
		vb1d[i] = (int) SvIV( *av_fetch(b1d, (I32) i, (I32) yes));
	};	
	THIS->chromfit(*beam, *orbit, vb1f, vb1d, mux, muy, method, numtries, tolerance, stepsize);

void
Teapot::map(vtps, beam, order)
	PacVTps* vtps
        BeamAttributes* beam
	int order
	CODE:
	THIS->map(*vtps, *beam, order);


void
Teapot::trackMap(vtps, beam, i1, i2)
	PacVTps* vtps
        BeamAttributes* beam
	int i1
	int i2
	CODE:
	THIS->trackMap(*vtps, *beam, i1, i2);

void
Teapot::transformOneTurnMap(output, oneTurnMap)
	PacVTps* output
	PacVTps* oneTurnMap	
	CODE:
	THIS->transformOneTurnMap(*output, *oneTurnMap);

void
Teapot::transformSectorMap(output, oneTurnMap, sectorMap)
	PacVTps* output
	PacVTps* oneTurnMap	
	PacVTps* sectorMap
	CODE:
	THIS->transformSectorMap(*output, *oneTurnMap, *sectorMap);


void
Teapot::matrix(vtps, beam, delta)
	PacVTps* vtps
        BeamAttributes* beam
	Position* delta
	CODE:
	THIS->matrix(*vtps, *beam, *delta);

void 
Teapot::decouple_(beam, orbit, ra11s, ra12s, ra13s, ra14s, rbfs, rbds, mux, muy)
	BeamAttributes* beam
	Position* orbit
	SV* ra11s
	SV* ra12s
	SV* ra13s
	SV* ra14s
	SV* rbfs
	SV* rbds
	double mux
	double muy
	CODE:
	AV* a11s  = (AV*) SvRV(ra11s);
	AV* a12s  = (AV*) SvRV(ra12s);
	AV* a13s = (AV*) SvRV(ra13s);
	AV* a14s = (AV*) SvRV(ra14s);
	AV* bfs   = (AV*) SvRV(rbfs);
	AV* bds   = (AV*) SvRV(rbds);
	int yes = 1, i;
	PacVector<int> va11s(1 + (int) av_len(a11s));
	for(i= 0; i < va11s.size(); i++) {
		va11s[i] = (int) SvIV( *av_fetch(a11s, (I32) i, (I32) yes));
	}
	PacVector<int> va12s(1 + (int) av_len(a12s));
	for(i= 0; i < va12s.size(); i++) {
		va12s[i] = (int) SvIV( *av_fetch(a12s, (I32) i, (I32) yes));
	}
	PacVector<int> va13s(1 + (int) av_len(a13s));
	for(i= 0; i < va13s.size(); i++) {
		va13s[i] = (int) SvIV( *av_fetch(a13s, (I32) i, (I32) yes));
	}
	PacVector<int> va14s(1 + (int) av_len(a14s));
	for(i= 0; i < va14s.size(); i++) {
		va14s[i] = (int) SvIV( *av_fetch(a14s, (I32) i, (I32) yes));
	}
	PacVector<int> vbfs(1 + (int) av_len(bfs));
	for(i= 0; i < vbfs.size(); i++) {
		vbfs[i] = (int) SvIV( *av_fetch(bfs, (I32) i, (I32) yes));
	}
	PacVector<int> vbds(1 + (int) av_len(bds));
	for(i= 0; i < vbds.size(); i++) {
		vbds[i] = (int) SvIV( *av_fetch(bds, (I32) i, (I32) yes));
	}
	THIS->decouple(*beam, *orbit, va11s, va12s, va13s, va14s, vbfs, vbds, mux, muy);


