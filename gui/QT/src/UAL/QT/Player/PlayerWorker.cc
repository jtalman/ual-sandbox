
#include "UAL/QT/Player/PlayerWorker.hh"
#include "UAL/QT/Player/PlayerShell.hh"
#include "UAL/QT/Player/BasicPlayer.hh"

UAL::QT::PlayerWorker::PlayerWorker()
{
  stopped = false;
  paused  = false;
}

void UAL::QT::PlayerWorker::run()
{
  m_turn = 0;
  // p_player->update(m_turn);   

  for(;;) {

    mutex.lock();
    if(stopped) {
      stopped = false;
      paused  = false;
      mutex.unlock();
      break;
    }
    if(paused) {
      continued.wait(&mutex);
      mutex.unlock();
      continue;
    }
    mutex.unlock();

    p_shell->run(m_turn);

    m_turn++;
    p_player->update(m_turn);   

    if(m_turn > p_player->getTurns()) return;
  }
}

void UAL::QT::PlayerWorker::initRun()
{
  if(!running()) {
    p_shell->initRun();
    p_player->update(0);
    mutex.lock();
    stopped = false;
    paused  = false;  
    mutex.unlock();
  }
}

void UAL::QT::PlayerWorker::startRun()
{
  if(!running()) {
    // p_shell->initRun();
    start();
  }
}

void UAL::QT::PlayerWorker::pauseRun()
{
  mutex.lock();
  paused = true;
  mutex.unlock();
}

void UAL::QT::PlayerWorker::continueRun()
{
  mutex.lock();
  paused = false;
  mutex.unlock();
  continued.wakeAll();
  
}

void UAL::QT::PlayerWorker::stopRun()
{
  mutex.lock();
  stopped = true;
  mutex.unlock();
}





