// Library       : UAL
// File          : UAL/APDF/APDF_Builder.cc
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#include <iostream>

#include "UAL/SMF/AcceleratorNodeFinder.hh"
#include "UAL/APF/PropagatorFactory.hh"
#include "UAL/APDF/APDF_Builder.hh"
#include "UAL/APDF/APDF_BuilderImpl.hh"


UAL::APDF_Builder::APDF_Builder()
{
}

UAL::APDF_Builder::~APDF_Builder()
{
}

void UAL::APDF_Builder::setBeamAttributes(const UAL::AttributeSet& ba)
{
  UAL::APDF_BuilderImpl::getInstance().setBeamAttributes(ba);
}

UAL::AcceleratorPropagator* UAL::APDF_Builder::parse(std::string& url)
{
  return UAL::APDF_BuilderImpl::getInstance().parse(url);
}

