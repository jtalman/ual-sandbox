// Library     : UAL
// File        : UAL/JVM/JavaSingleton.cc
// Copyright   : see Copyright file
// Authors     : N.Malitsky

#include <iostream>
#include "UAL/JVM/JavaSingleton.hh"  

UAL::JavaSingleton* UAL::JavaSingleton::s_instance = 0;

UAL::JavaSingleton::JavaSingleton()
{
}

UAL::JavaSingleton::~JavaSingleton()
{
}

UAL::JavaSingleton* UAL::JavaSingleton::getInstance()
{
  if(s_instance == 0){
    s_instance = new UAL::JavaSingleton();
  }
  return s_instance;
}

long UAL::JavaSingleton::create(JavaVMOption* options, int nOptions)
{ 

  JavaVMInitArgs vmargs;
  vmargs.version = JNI_VERSION_1_2;
  vmargs.options = options;
  vmargs.nOptions = 2;
  vmargs.ignoreUnrecognized = JNI_FALSE;

  long result = JNI_CreateJavaVM(&m_jvm,(void **)&m_jnienv, &vmargs);
  if(result == JNI_ERR) {
    std::cerr << "Error invoking the JVM" << std::endl;
    result = 0;
  }
  return result;
}

JavaVM* UAL::JavaSingleton::getJavaVM(){
  return m_jvm;
}

JNIEnv* UAL::JavaSingleton::getJNIEnv(){
  return m_jnienv;
}

void UAL::JavaSingleton::destroy()
{ 
  if(m_jvm != 0){
    m_jvm->DestroyJavaVM();
  }
}




