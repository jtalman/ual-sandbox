// Library       : TIBETAN
// File          : TIBETAN/JFreeChart/RampWindowController.cc
// Copyright     : see Copyright file
// Author        : J.Wei
// C++ version   : N.Malitsky 

#include <iostream>
#include "UAL/JVM/JavaSingleton.hh"
#include "TIBETAN/JFreeChart/JRampWindowController.hh"

TIBETAN::JRampWindowController::JRampWindowController() 
{
  // Get Java VM

  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  JNIEnv* jenv = js->getJNIEnv();

  // Get the Java class
  jclass cls = jenv->FindClass("ual/tibetan/jfreechart/RampWindowController");
  if( cls == NULL ) {
    printf("can't find class ual.tibetan.jfreechart.RampWindowController\n");
    return;
  }

  // jenv->ExceptionClear();  

  // Allocate the java object
  jmethodID mid = jenv->GetMethodID(cls, "<init>", "()V");
  jobject lref = jenv->NewObject(cls, mid);
  m_jobject = jenv->NewGlobalRef(lref);

  // Define method ids
  m_initWindow_id = jenv->GetMethodID(cls, "initWindow", "()V");
  m_showWindow_id = jenv->GetMethodID(cls, "showWindow", "()V");
  m_updateData_id = jenv->GetMethodID(cls, "updateData", "()V");
  m_setTimeRange_id = jenv->GetMethodID(cls, "setTimeRange", "(DD)V");
  m_setGammaRange_id = jenv->GetMethodID(cls, "setGammaRange", "(DD)V");
  m_addGammaValue_id = jenv->GetMethodID(cls, "addGammaValue", "(DD)V"); 

  m_jclass = jenv->NewGlobalRef(cls);   
     
}

TIBETAN::JRampWindowController::~JRampWindowController() 
{
  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  JNIEnv* jenv = js->getJNIEnv();
  jenv->DeleteGlobalRef(m_jobject);
  jenv->DeleteGlobalRef(m_jclass);
}

void TIBETAN::JRampWindowController::initWindow()
{
  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  JNIEnv* jenv = js->getJNIEnv();
  jenv->CallObjectMethod(m_jobject, m_initWindow_id);
}

void TIBETAN::JRampWindowController::showWindow()
{
  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  JNIEnv* jenv = js->getJNIEnv();
  jenv->CallObjectMethod(m_jobject, m_showWindow_id);
}

void TIBETAN::JRampWindowController::updateData()
{
  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  JNIEnv* jenv = js->getJNIEnv();
  jenv->CallObjectMethod(m_jobject, m_updateData_id);
}

void TIBETAN::JRampWindowController::setTimeRange(double minTime, double maxTime)
{
  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  JNIEnv* jenv = js->getJNIEnv();
  jenv->CallObjectMethod(m_jobject, m_setTimeRange_id, minTime, maxTime);
}

void TIBETAN::JRampWindowController::setGammaRange(double minGamma, double maxGamma)
{
  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  JNIEnv* jenv = js->getJNIEnv();
  jenv->CallObjectMethod(m_jobject, m_setGammaRange_id, minGamma, maxGamma);
}

void TIBETAN::JRampWindowController::addGammaValue(double t, double gamma)
{
  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  JNIEnv* jenv = js->getJNIEnv();
  jenv->CallObjectMethod(m_jobject, m_addGammaValue_id, t, gamma);
}
