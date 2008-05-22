// Library       : UAL
// File          : UAL/APDF/APDF_BuilderImpl.cc
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#include <iostream>

#include "UAL/SMF/AcceleratorNodeFinder.hh"
#include "UAL/APF/PropagatorFactory.hh"

#include "UAL/APDF/APDF_BuilderImpl.hh"
// #include "UAL/APDF/APDF_CreateElement.hh"

UAL::APDF_BuilderImpl* UAL::APDF_BuilderImpl::s_theInstance = 0;

UAL::APDF_BuilderImpl::APDF_BuilderImpl()
{
}

UAL::APDF_BuilderImpl::~APDF_BuilderImpl()
{
}

UAL::APDF_BuilderImpl& UAL::APDF_BuilderImpl::getInstance()
{
  if(s_theInstance == 0) {
    s_theInstance = new UAL::APDF_BuilderImpl();
  }
  return *s_theInstance;
}

void UAL::APDF_BuilderImpl::setBeamAttributes(const UAL::AttributeSet& ba)
{
  // Copy data
  UAL::RCIPtr<UAL::AttributeSet> baPtr(ba.clone());

  m_ba = baPtr;
}

UAL::AcceleratorPropagator* UAL::APDF_BuilderImpl::parse(std::string& url)
{
  UAL::AcceleratorPropagator* ap = 0;

  xmlDocPtr doc = xmlParseFile(url.c_str());
  if(doc == 0) {
    return ap;
  }

  xmlNodePtr rootNode = xmlDocGetRootElement(doc);
  if(rootNode == 0){
    xmlFreeDoc(doc);
    return ap;
  }
 
  xmlNodePtr propagatorNode = getXmlNode(rootNode, "propagator");
  if(propagatorNode == 0){
    xmlFreeDoc(doc);
    return ap;
  }  

  ap = createAP(propagatorNode);
  if(ap == 0){
    xmlFreeDoc(doc);
    return ap;
  }

  xmlNodePtr createNode = getXmlNode(propagatorNode, "create");
  if(createNode == 0){
    xmlFreeDoc(doc);
    return ap;
  }  

  UAL::APDF_CreateElement createElement;
  createElement.init(createNode);

  createLinks(ap, createElement);
   
  xmlFreeDoc(doc);  
  return ap;
}

void UAL::APDF_BuilderImpl::setLattice(const std::string& accName){

  UAL::AcceleratorNodeFinder::Iterator it = 
    UAL::AcceleratorNodeFinder::getInstance().find(accName);

  if(it == UAL::AcceleratorNodeFinder::getInstance().end()){
    std::cout << "ADBF Builder lattice " << accName << " has not been found " << std::endl;
  }
  else {
    m_lattice = (it->second).operator->();
  }

  double at = 0;
  int lsize = m_lattice->getNodeCount();

  for(int i=0; i < lsize; i++){

    UAL::AcceleratorNode* const anode = m_lattice->getNodeAt(i);

    std::string elname = anode->getDesignName();
    std::string eltype = anode->getType();

    anode->setPosition(at);

    // std::cout << i << "at = " << at << " name = " << elname << ", type = " << eltype << std::endl;

    at += anode->getLength();
  }

}


void UAL::APDF_BuilderImpl::createLinks(UAL::AcceleratorPropagator* ap, 
				    UAL::APDF_CreateElement& createElement)
{

  std::list<PropagatorNodePtr> sectorLinks;
  // makeSectorLinks(sectorLinks, createElement);

  std::list<PropagatorNodePtr> elementLinks;
  makeElementLinks(elementLinks, createElement);

  std::list<PropagatorNodePtr> typeLinks;
  makeTypeLinks(typeLinks, createElement);

  std::list<PropagatorNodePtr> mergedLinks;
  // mergeLinks(sectorLinks, elementLinks, typeLinks, mergedLinks);

  // std::cout << "createLinks " << std::endl;
  UAL::APDF_LinkElement defaultSectorLink = createElement.selectSectorLink("Default");
  if(defaultSectorLink.getType() != UAL::APDF_LinkElement::EMPTY) {
    // std::cout << "size of element links " << elementLinks.size() << std::endl; 
    addDefaultSectorLink(elementLinks, defaultSectorLink, ap);
    return;
  }

  UAL::APDF_LinkElement defaultTypeLink = createElement.selectTypeLink("Default");
  if(defaultTypeLink.getType() != UAL::APDF_LinkElement::EMPTY) {
    // std::cout << "size of type links " << typeLinks.size() << std::endl;
    addDefaultTypeLink(typeLinks, defaultTypeLink, ap);
    return;
  }
}

void UAL::APDF_BuilderImpl::addDefaultTypeLink(std::list<PropagatorNodePtr>& links,
					       UAL::APDF_LinkElement& defaultTypeLink,
					       UAL::AcceleratorPropagator* ap)
{

  UAL::PropagatorSequence& rootNode = ap->getRootNode();

  UAL::PropagatorNodePtr defaultTypePropagator = defaultTypeLink.m_algPtr;

  // Start the loop over lattice elements

  int is0   = 0;
  int is1   = 0;
  int lsize = m_lattice->getNodeCount();

  std::string frontName, backName;
  std::list<PropagatorNodePtr>::iterator it;

  for(it = links.begin(); it != links.end(); it++) {

    int i;
    
    // find is1
    frontName = (*it)->getFrontAcceleratorNode().getDesignName();
    // std::cout << "front name : " << frontName << std::endl;
    for(i = is0; i < lsize; i++){
      UAL::AcceleratorNode* const anode = m_lattice->getNodeAt(i);
      is1 = i;

      std::string elname = anode->getDesignName();
      if(!frontName.compare(elname)) {
	break;
      }

      UAL::PropagatorNode* propNode = (UAL::PropagatorNode*) defaultTypePropagator->clone();
      propNode->setLatticeElements(*m_lattice, i, i, *m_ba);
 
      UAL::PropagatorNodePtr propNodePtr(propNode);   
      rootNode.add(propNodePtr);

    }

    // add link propagator
    rootNode.add(*it);

    // find is0
    backName = (*it)->getBackAcceleratorNode().getDesignName();
    // std::cout << "back name : " << backName << std::endl;
    for(i = is1; i < lsize; i++){
      UAL::AcceleratorNode* const anode = m_lattice->getNodeAt(i);
      is0 = i;
      std::string elname = anode->getDesignName();
      if(!backName.compare(elname)) {
	break;
      }
    }

    if(is0 == is1) is0 = is1 + 1;

    if(i == lsize) {
      std::cout << "APDF_BuilderImpl: there is no element " << backName << std::endl;
      exit(1);
    }
    
  }

  if(lsize != is0){

    for(int j = is0; j < lsize; j++){

      UAL::PropagatorNode* propNode = (UAL::PropagatorNode*) defaultTypePropagator->clone();
      propNode->setLatticeElements(*m_lattice, j, j, *m_ba);
 
      UAL::PropagatorNodePtr propNodePtr(propNode);   
      rootNode.add(propNodePtr);

    }
   
  } 
}

void UAL::APDF_BuilderImpl::addDefaultSectorLink(std::list<PropagatorNodePtr>& links,
					     UAL::APDF_LinkElement& defaultSectorLink,
					     UAL::AcceleratorPropagator* ap)
{

  UAL::PropagatorSequence& rootNode = ap->getRootNode();

  UAL::PropagatorNodePtr defaultSectorPropagator = defaultSectorLink.m_algPtr;

  // Start the loop over lattice elements

  int is0   = 0;
  int is1   = 0;
  int lsize = m_lattice->getNodeCount();

  std::string frontName, backName;
  std::list<PropagatorNodePtr>::iterator it;

  for(it = links.begin(); it != links.end(); it++) {

    int i;
    
    // find is1
    //  std::cout << "before finding is1. is0=" << is0 << ", is1=" << is1 << " " << lsize << std::endl;
    frontName = (*it)->getFrontAcceleratorNode().getDesignName();
    for(i = is0; i < lsize; i++){
      UAL::AcceleratorNode* const anode = m_lattice->getNodeAt(i);
      is1 = i;

      std::string elname = anode->getDesignName();
      if(!frontName.compare(elname)) {
	break;
      }
    }
    // std::cout << "after finding is1. is0=" << is0 << ", is1=" << is1 << " " << lsize << std::endl;

    // add default propagator    
    if(is1 != is0) {
 
      UAL::PropagatorNode* sectorNode = (UAL::PropagatorNode*) defaultSectorPropagator->clone();
      sectorNode->setLatticeElements(*m_lattice, is0, is1, *m_ba);
 
      UAL::PropagatorNodePtr sectorNodePtr(sectorNode);   
      rootNode.add(sectorNodePtr);
    }

    // std::cout << "after adding default propagator " << std::endl;

    // add link propagator
    rootNode.add(*it);

    // std::cout << "after adding node" << std::endl;
    // std::cout << "front name= " << (*it)->getFrontAcceleratorNode().getDesignName() << std::endl;
    // std::cout << "back name= " << (*it)->getBackAcceleratorNode().getDesignName() << std::endl;

    // std::cout << is0 << " " << is1 << " " << lsize << std::endl;

    // find is0
    // std::cout << "before finding is0. is0=" << is0 << ", is1=" << is1 << " " << lsize << std::endl;
    backName = (*it)->getBackAcceleratorNode().getDesignName();
    for(i = is1; i < lsize; i++){
      UAL::AcceleratorNode* const anode = m_lattice->getNodeAt(i);
      is0 = i;
      std::string elname = anode->getDesignName();
      if(!backName.compare(elname)) {
	break;
      }
    }
    // std::cout << "after finding is0. is0=" << is0 << ", is1=" << is1 << " " << lsize << std::endl;


    if(is0 == is1) is0 = is1 + 1;

    if(i == lsize) {
      std::cout << "APDF_BuilderImpl: there is no element " << backName << std::endl;
      exit(1);
    }
    
  }

  // std::cout << "after loop. is0=" << is0 << ", is1=" << is1 << " " << lsize << std::endl;

  if(lsize != is0){
   
    // std::cout << "before cloning the last sector " << std::endl;

    // Create and add sector propagator
    UAL::PropagatorNode* sectorNode = 
      (UAL::PropagatorNode*) defaultSectorPropagator->clone();
    // std::cout << "after cloning the last sector " << std::endl;

    sectorNode->setLatticeElements(*m_lattice, is0, lsize, *m_ba);
    // std::cout << "after setting the last sector " << std::endl;
       
    // delegate ownership to the smart pointer
    UAL::PropagatorNodePtr sectorNodePtr(sectorNode);   
    rootNode.add(sectorNodePtr);  
    // std::cout << "after adding the last sector " << std::endl;   
  } 
}



void UAL::APDF_BuilderImpl::makeElementLinks(std::list<PropagatorNodePtr>& elementLinks, 
					 UAL::APDF_CreateElement& createElement)
{
 
  // Start the loop over lattice elements

  double l  = 0;
  int lsize = m_lattice->getNodeCount();

  for(int i=0; i < lsize; i++){

    UAL::AcceleratorNode* const anode = m_lattice->getNodeAt(i);

    std::string elname = anode->getDesignName();
    std::string eltype = anode->getType();
    l += anode->getLength(); 
 
    // std::cout << i << " name = " << elname << ", type = " << eltype << std::endl;

    // 1. Select propagator
    UAL::APDF_LinkElement&  link = createElement.selectElementLink(elname);
    if(link.getType() == UAL::APDF_LinkElement::EMPTY) continue;

    // std::cout << "element link " << i << " name = " << elname << ", type = " << eltype << std::endl;
      
    UAL::PropagatorNode* node = link.m_algPtr->clone();

    // 2. Initialize it
    node->setLatticeElements(*m_lattice, i, i, *m_ba);

    // 3. Delegate ownership to the smart pointer
    UAL::PropagatorNodePtr nodePtr(node);  

    // 4. Add the selected propagator to the list of element links
    elementLinks.push_back(nodePtr); 
  }


}

void UAL::APDF_BuilderImpl::makeTypeLinks(std::list<PropagatorNodePtr>& typeLinks, 
				      UAL::APDF_CreateElement& createElement)
{
 
  // Start the loop over lattice elements

  double l  = 0;
  int lsize = m_lattice->getNodeCount();

  for(int i=0; i < lsize; i++){

    UAL::AcceleratorNode* const anode = m_lattice->getNodeAt(i);

    std::string elname = anode->getDesignName();
    std::string eltype = anode->getType();
    l += anode->getLength(); 
 
    // std::cout << i << " name = " << elname << ", type = " << eltype << std::endl;

    // 1. Select propagator
    UAL::APDF_LinkElement&  link = createElement.selectTypeLink(eltype);
    if(link.getType() == UAL::APDF_LinkElement::EMPTY) continue;

    // std::cout << "type link " << i << " name = " << elname << ", type = " << eltype << std::endl;
      
    UAL::PropagatorNode* node = link.m_algPtr->clone();

    // 2. Initialize it
    node->setLatticeElements(*m_lattice, i, i, *m_ba);

    // std::cout << "front name " << node->getFrontAcceleratorNode().getDesignName() << std::endl;

    // 3. Delegate ownership to the smart pointer
    UAL::PropagatorNodePtr nodePtr(node);  

    // 4. Add the selected propagator to the list of element links
    typeLinks.push_back(nodePtr); 
  }
}

xmlNodePtr UAL::APDF_BuilderImpl::getXmlNode(xmlNodePtr parentNode, const char* tag)
{
  xmlNodePtr cur = parentNode->xmlChildrenNode;

  while(cur != 0){
    if ((!xmlStrcmp(cur->name, (const xmlChar *) tag))){
      break;
    }
    cur = cur->next;
  }
  return cur;  
}

UAL::AcceleratorPropagator* UAL::APDF_BuilderImpl::createAP(xmlNodePtr apNode)
{
  UAL::AcceleratorPropagator* ap = 0;

  xmlChar* accName = xmlGetProp(apNode, (const xmlChar *) "accelerator");
  if(accName == 0) return ap;

  setLattice((const char*) accName);
  xmlFree(accName);

  if(m_lattice == 0) {
    return ap;
  }
  // std::cout << "accelerator = " << m_lattice->getName() << std::endl;    

  ap = new UAL::AcceleratorPropagator();

  xmlChar* apName = xmlGetProp(apNode, (const xmlChar *) "id");
  if(apName != 0) {
    ap->setName((const char*) apName);
    xmlFree(apName);
  }

  // std::cout << "id = " << ap->getName() << std::endl;  
 
  return ap;
}




