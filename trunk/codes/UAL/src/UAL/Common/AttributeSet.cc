// Library     : UAL
// File        : UAL/Common/AttributeSet.cc
// Copyright   : see Copyright file
// Authors     : N.Malitsky & R.Talman

#include "UAL/Common/AttributeSet.hh"

std::vector<std::string> UAL::AttributeSet::s_emptyVector;

UAL::AttributeSet::AttributeSet()
{
}

UAL::AttributeSet::~AttributeSet()
{
}

double UAL::AttributeSet::getAttribute(const std::string& attrName) const
{
  return 0;
}

void UAL::AttributeSet::setAttribute(const std::string& , double)
{
}

const std::vector<std::string>& UAL::AttributeSet::getAttributeNames() const
{
  return s_emptyVector;
}

UAL::AttributeSet* UAL::AttributeSet::clone() const
{
  return new UAL::AttributeSet();
}


