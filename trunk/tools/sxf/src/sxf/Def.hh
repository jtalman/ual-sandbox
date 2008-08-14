//# Library     : SXF
//# File        : Def.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_DEF_H
#define SXF_DEF_H

#include <string>
#include <string.h>

using namespace std;



/** 
    The SXF represents a flat, complete, and independent description 
      of the current accelerator state, authors: H.Grote, J.Holt, N.Malitsky, 
      F.Pilat, R.Talman, C.G.Trahern, W.Fischer.

      SXF includes the full-instantiated view of the accelerator and  provides the mechanism 
      for its integration with the present site-specific design hierarchical models. 
      This approach stems from the desire to make this format neutral to different 
      conceptual models and adaptable to arbitrary data stores. The SXF can be 
      considered as the additional independent layer to existing design data structures, 
      particular to the Standard Input Format (SIF) <i>design</i> components, 
      <i>element</i> and <i>beam line. </i>The relationship between SIF and SXF 
      structures is provided by the references embedded in the SXF objects. The 
      SXF format is based on the integration of the MAD <i>sequence </i>and UAL/SMF 
      <i>element buckets</i>, orthogonal collection of element attributes. It 
      preserves all SIF element types and provides the mechanism for introducing 
      new ones, such as CESR superimposed solenoid and quadrupole elements, LHC 
      and CESR parasitic beam-beam effects, Muon Collider Ionization Cooling, 
      and others. 
 
 */

namespace SXF {
}

#define SXF_FALSE 0
#define SXF_TRUE  1

// SXF-specific structure that is used in the GPERF, a perfect hash 
// function generator

struct SXF_Key {
  char* name;
  int   number;
};

#endif
