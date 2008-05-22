#include <iostream>
#include <string>
#include <jni.h>
#include "UAL/JVM/JWindowManager.hh"


int main(int argc, char *argv[], char **envp) {

  // Define the Java class path

  char* ualhome = getenv("UAL");
  std::string strClasspath = "-Djava.class.path=";
  strClasspath += ualhome;
  strClasspath += "/codes/UAL/lib/java/ualcore.jar";
  char* charClasspath = new char[strClasspath.size() + 1];
  strcpy(charClasspath, strClasspath.data());

  // Define Java VM attributes

  JavaVMOption options[2];
  options[0].optionString = "-Djava.compiler=NONE";
  options[1].optionString = charClasspath;

  // Create Java VM

  UAL::JavaSingleton* js = UAL::JavaSingleton::getInstance();
  js->create(options, 2);

  std::cout << "Java starts" << std::endl;

  // Create C++ proxy to the Java Window Manager

  UAL::JWindowManager wm;
  wm.initMainWindow();

  // Shows the Main Window 

  wm.showMainWindow();

  // Start the infinite loop

  while(1){
  }

  // js->destroy();

}
