// Library     : UAL
// File        : UAL/JVM/JavaSingleton.hh
// Copyright   : see Copyright file
// Authors     : N.Malitsky 

#ifndef UAL_JAVA_SINGLETON_HH
#define UAL_JAVA_SINGLETON_HH

#include <jni.h>
#include "UAL/Common/Object.hh"

namespace UAL {

  /**
   * A singleton of the Java Virtual Machine (VM) .
   */

  class JavaSingleton : public Object
  {
  public:

    /** Returns a pointer to the only instance of this class */
    static JavaSingleton* getInstance();

    /** Creates Java VM */
    long create(JavaVMOption* options, int nOptions);

    /** Destroy Java VM */
    void destroy();

    /** Returns a pointer to Java VM */
    JavaVM* getJavaVM();

    /** Returns a pointer to Java JNI environment */
    JNIEnv* getJNIEnv();

  protected:

    /** Singleton */
    static JavaSingleton* s_instance;

    /** Java VM */
    JavaVM* m_jvm;

    /** JNI Environment */
    JNIEnv *m_jnienv;

  protected:

    /** Destructor */
    virtual ~JavaSingleton();

  private:
      
    /** Constructor */
    JavaSingleton();

  };

};

#endif
