#ifndef UAL_QT_PLAYER_WORKER_HH
#define UAL_QT_PLAYER_WORKER_HH

#include <map>

#include <qthread.h>

namespace UAL
{
  namespace QT {
  class PlayerShell;
  class BasicPlayer;

  class PlayerWorker : public QThread
  {
  public:

    /** Constructor */
    PlayerWorker();

    void setPlayer(BasicPlayer* player) { p_player = player; }
    void setShell(PlayerShell* shell) { p_shell = shell; }

    void run();

    void initRun();
    void startRun();
    void pauseRun();
    void continueRun();
    void stopRun();

  protected:

    PlayerShell* p_shell;    
    BasicPlayer* p_player;

    QMutex mutex;
    volatile bool stopped;

    QWaitCondition continued;
    volatile bool paused;

    int m_turn;

  };
 }
}

#endif
