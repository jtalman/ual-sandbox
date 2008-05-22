// Library     : PAC
// File        : SMF/PacSmf.cc
// Copyright   : see Copyright file
// Description : Standard Machine Format.
// Author      : Nikolay Malitsky


#include <stdio.h>
#include "SMF/PacSmf.h"

// Constructor
PacSmf::PacSmf() { }

// Return the extent of design elements;
PacGenElements* PacSmf::elements() { return PacGenElements::instance(); }

// Return the extent of design beam lines;
PacLines* PacSmf::lines() { return PacLines::instance(); }

// Return the extent of real lattices;
PacLattices* PacSmf::lattices() { return PacLattices::instance(); }

// Return the extent of elemenet keys;
PacElemKeys* PacSmf::elemKeys() { return PacElemKeys::instance(); }

// Return the extent of elemenet bucket keys;
PacElemBucketKeys* PacSmf::bucketKeys() { return PacElemBucketKeys::instance(); }

