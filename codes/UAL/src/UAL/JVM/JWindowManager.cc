// Library       : UAL
// File          : UAL/JVM/JWindowManager.cc
// Copyright     : see Copyright file
// Author        : J.Wei
// C++ version   : N.Malitsky 

#include "UAL/JVM/JWindowManager.hh"

UAL::JWindowManager::JWindowManager() 
{
  // Get Java VM

  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  JNIEnv* jenv = js->getJNIEnv();

  // Get the Java class
  jclass cls = jenv->FindClass("ual/gui/WindowManager");
  if( cls == NULL ) {
    printf("can't find class ual.gui.WindowManager\n");
    return;
  }

  jenv->ExceptionClear();  

  // Allocate the java object
  jobject lref = jenv->AllocObject(cls);
  m_jobject = jenv->NewGlobalRef(lref);

  // Define method ids
  m_initMainWindow_id = jenv->GetMethodID(cls, "initMainWindow", "()V");
  m_showMainWindow_id = jenv->GetMethodID(cls, "showMainWindow", "()V");

  m_jclass = jenv->NewGlobalRef(cls);   
}

UAL::JWindowManager::~JWindowManager() 
{
  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  JNIEnv* jenv = js->getJNIEnv();
  jenv->DeleteGlobalRef(m_jobject);
  jenv->DeleteGlobalRef(m_jclass);
}

void UAL::JWindowManager::initMainWindow()
{
  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  JNIEnv* jenv = js->getJNIEnv();
  jenv->CallObjectMethod(m_jobject, m_initMainWindow_id);
}

void UAL::JWindowManager::showMainWindow()
{
  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  JNIEnv* jenv = js->getJNIEnv();
  jenv->CallObjectMethod(m_jobject, m_showMainWindow_id);
}

void UAL::JWindowManager::hideMainWindow()
{
  // UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  // JNIEnv* jenv = js->getJNIEnv();
  // jenv->CallObjectMethod(m_jobject, m_hideWindow_id);
}
