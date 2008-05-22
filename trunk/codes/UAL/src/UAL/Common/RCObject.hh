// Library     : UAL
// File        : UAL/Common/PacRCObject.hh
// Copyright   : see Copyright file
// Authors     : implemented after Scott Meyers' approach ("More effective C++")

#ifndef UAL_RCOBJECT_HH
#define UAL_RCOBJECT_HH

namespace UAL {

  /** An abstract basis class of reference-counted objects.
   */

  class RCObject {

  public :

    /** Constructor */
    RCObject();

    /** Copy constructor */
    RCObject(const RCObject& rhs);

    /** Copy operator */
    RCObject& operator=(const RCObject& rhs);

    /** Destructor */
    virtual ~RCObject();

    /**  Augments a reference counter. */
    int addReference();

    /** Dicrements a reference counter.*/
    int removeReference();

    /** Makes this object unshreable. */
    void markUnshareable();

    /** Checks if this object is shareable. */
    bool isShareable() const;

    /** Checks if this object is shared by other objects (reference counter is not 0). */
    bool isShared() const;

  private:

    int refCount;
    
    bool shareable;

  };


}

#endif
