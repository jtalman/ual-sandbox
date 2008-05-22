// Library       : AIM
// File          : AIM/BTF/BTFSignal.cc
// Copyright     : see Copyright file

#include "AIM/BTF/BTFSignal.hh"

AIM::BTFSignal::BTFSignal()
{
}

void AIM::BTFSignal::resize(int size)
{
  cts.resize(size);
  density.resize(size);
  xs.resize(size);
  ys.resize(size);  
}
