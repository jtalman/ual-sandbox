//# Library     : SXF
//# File        : echo_sxf/EchoElemBucket.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef SXF_ECHO_ELEM_BUCKET_H
#define SXF_ECHO_ELEM_BUCKET_H

#include "echo_sxf/EchoDef.hh"

namespace SXF {

  /**
   * Implements an echo writer of an element bucket 
   * with scalar and array attributes.
   */

  class EchoElemBucket : public ElemBucket
  {
  public:

    /** Constructor. */
    EchoElemBucket(OStream& out, const char* type, ElemBucketHash* hash);

    /** Do nothing, Return true. */
    int openObject(const char*);

    /** Do nothing. */
    void update();

    /** Do nothing. */
    void close();

    /** Do nothing, Return true. */
    int openArray();

    /** Do nothing. */
    void closeArray();

    /** Do nothing, Return true */
    int openHash();

    /** Do nothing. */
    void closeHash();

    /** Do nothing, Return true. */
    int setScalarValue(double value);

    /** Do nothing, Return true. */
    int setArrayValue(double value);

    /** Do nothing, Return true. */
    int setHashValue(double value, int index);

  };
}

#endif
