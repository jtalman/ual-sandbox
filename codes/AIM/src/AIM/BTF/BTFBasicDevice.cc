// Library       : AIM
// File          : AIM/BTF/BTFBasicDevice.cc
// Copyright     : see Copyright file

#include "UAL/SMF/AcceleratorNodeFinder.hh"
#include "AIM/BTF/BTFBasicDevice.hh"


void AIM::BTFBasicDevice::setLatticeElements(const UAL::AcceleratorNode& sequence, 
					     int i0, int i1, 
					     const UAL::AttributeSet& attSet)
{
  if(i0 < sequence.getNodeCount()) m_frontNode = sequence.getNodeAt(i0);
  if(i1 < sequence.getNodeCount()) m_backNode  = sequence.getNodeAt(i1);

}

UAL::AcceleratorNode& AIM::BTFBasicDevice::getFrontAcceleratorNode()
{
  if(m_frontNode == 0) return UAL::AcceleratorNodeFinder::getEmptyNode();
  else return *m_frontNode;
}

UAL::AcceleratorNode& AIM::BTFBasicDevice::getBackAcceleratorNode()
{
  if(m_backNode == 0) return UAL::AcceleratorNodeFinder::getEmptyNode();
  else return *m_backNode;
}

