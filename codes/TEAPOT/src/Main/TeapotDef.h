// Library     : TEAPOT
// File        : Teapot/TeapotDef.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef TEAPOT_DEF_H
#define TEAPOT_DEF_H

using namespace std;

#include <assert.h>
#include <iostream>
#include <math.h>

#include <string>

// PAC

#include "PAC/Common/PacException.h"

/**
   Thin Element Accelerator Program for Optics and Tracking (collection of symplectic trackers and mappers, 
   collection of correction algorithms), authors: Richard Talman and Lindsay Schachinger
   <p> (under construction).
 */
namespace TEAPOT {
}

// General constants

#ifndef PI
#define PI   3.1415926535898
#endif

#ifndef BEAM_CLIGHT
#define BEAM_CLIGHT   2.99792458e+8
#endif

// Teapot constans

#define TEAPOT_ORDER 11
#define TEAPOT_APERTURE 1.
#define TEAPOT_DIMENSION 6
#define TEAPOT_EPS 1.0e-12

#endif
