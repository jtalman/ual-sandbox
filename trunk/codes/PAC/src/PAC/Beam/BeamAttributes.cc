// Program     : PAC
// File        : PAC/Beam/BeamAttributes.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "UAL/Common/Def.hh"
#include "PAC/Common/PacException.h"
#include "PAC/Beam/BeamAttributes.hh"

// Constructor 
PAC::BeamAttributes::BeamAttributes() {
  initialize();
}

// Copy constructor
PAC::BeamAttributes::BeamAttributes(const PAC::BeamAttributes& ba) {
  define(ba);
}

// Destructor
PAC::BeamAttributes::~BeamAttributes(){
}

// Copy operator

const PAC::BeamAttributes& PAC::BeamAttributes::operator=(const PAC::BeamAttributes& ba) {
  if(this == &ba) return *this;
  define(ba);
  return *this;
}

UAL::AttributeSet* PAC::BeamAttributes::clone() const
{
  return new PAC::BeamAttributes(*this);
}

// Access methods

double PAC::BeamAttributes::getMass() const {
  return m_mass;
}

void PAC::BeamAttributes::setMass(double m)
{
  if(m <= 0.0 || m > getEnergy()) {
    std::string msg = "Error : PAC::BeamAttributes::setMass(double m) : m is out of [0.0, energy] \n";
    PacDomainError(msg).raise();
  }
  m_mass = m;
}

double PAC::BeamAttributes::getEnergy() const {
  return m_energy;
}

void PAC::BeamAttributes::setEnergy(double e)
{
  if(e < getMass()) {
    std::string msg = "Error : PAC::BeamAttributes::setEnergy(double e) : e < mass \n";
    PacDomainError(msg).raise();
  }
  m_energy = e;
}

double PAC::BeamAttributes::getCharge() const {
  return m_charge; 
}

void PAC::BeamAttributes::setCharge(double c) {
  m_charge = c;
}

double PAC::BeamAttributes::getElapsedTime() const {
  return m_time;
}

void PAC::BeamAttributes::setElapsedTime(double t) {
  m_time = t;
}

double PAC::BeamAttributes::getRevfreq() const {
  return m_revfreq;
}

void PAC::BeamAttributes::setRevfreq(double f) {
  m_revfreq = f;
}

double PAC::BeamAttributes::getMacrosize() const {
   return m_macrosize;
}

void PAC::BeamAttributes::setMacrosize(double ms) {
   m_macrosize = ms;
}

double PAC::BeamAttributes::getG() const {
   return m_G;
}

void PAC::BeamAttributes::setG(double g) {
   m_G = g;
}

// Private methods

void PAC::BeamAttributes::initialize()
{
  m_energy = UAL::infinity;
  m_mass = UAL::pmass;
  m_charge = 1.0;
  m_revfreq = 0.0;
  m_macrosize = 0.0;
  m_G = UAL::pG;
}

void PAC::BeamAttributes::define(const PAC::BeamAttributes& ba)
{
  m_energy = ba.m_energy;
  m_mass = ba.m_mass;
  m_charge = ba.m_charge;
  m_revfreq = ba.m_revfreq;
  m_macrosize = ba.m_macrosize;
}


