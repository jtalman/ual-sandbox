// Program     : PAC
// File        : PAC/Common/PacException.cc
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#include "PAC/Common/PacException.h"

// Constructor

PacException::PacException(){
}

PacException::PacException(const std::string& msg) 
  : desc(msg) {
}

// Destructor
PacException::~PacException(){
}

// 
void PacException::raise() {
   std::cerr << desc << "\n"; 
   do_raise(); 
}

const std::string& PacException::what() const {
  return desc;
}

void PacException::do_raise() {
  _PAC_RAISE(*this); 
}

PacLogicError::PacLogicError(const std::string& msg)
  : PacException(msg) {
}

PacRuntimeError::PacRuntimeError(const std::string& msg)
  : PacException(msg) {
}

PacDomainError::PacDomainError(const std::string& msg) 
  : PacLogicError(msg) {
}

void PacDomainError::do_raise() {
  _PAC_RAISE(*this);
}

PacAllocError::PacAllocError(const std::string& msg) 
  : PacRuntimeError(msg) {
}

void PacAllocError::do_raise() {
  _PAC_RAISE(*this);
}



