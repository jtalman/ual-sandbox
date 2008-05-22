// Library     : PAC
// File        : PAC/Common/PacException.h
// Copyright   : see Copyright file
// Description : Exception library (a wrapper and extension of the standard C++ library)
// Author      : Nikolay Malitsky 

#ifndef PAC_EXCEPTION_H
#define PAC_EXCEPTION_H

#include <iostream>
#include <string>

// #include <stdexcept> -- conflicts with .../2.7.2/include/math.h
// #define _PAC_RAISE(a) throw(a) -- conflicts with optimization -O for gcc 2.7.2 version

#include <assert.h>
#define _PAC_RAISE(a) assert(0)

class PacException
{
public:

  PacException(const std::string& msg);

  virtual ~PacException();

  void raise();

  virtual const std::string& what() const;

protected:

  PacException();
  virtual void do_raise();

private:

  std::string desc;
};

class PacLogicError : public PacException
{
public:
  PacLogicError(const std::string& msg);
};

class PacRuntimeError : public PacException
{
public:
  PacRuntimeError(const std::string& msg);
};

class PacDomainError : public PacLogicError 
{
public:

  PacDomainError(const std::string&  msg);

private:

  void do_raise();

};

class PacAllocError : public PacRuntimeError
{
public:

  PacAllocError(const std::string& msg);

private:

  void do_raise();

};

#endif
