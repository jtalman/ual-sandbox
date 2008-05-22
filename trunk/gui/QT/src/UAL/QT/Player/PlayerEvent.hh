#ifndef UAL_QT_PLAYER_EVENT_HH
#define UAL_QT_PLAYER_EVENT_HH

#include <qevent.h>

namespace UAL
{
 namespace QT {
  class PlayerEvent : public QCustomEvent
  {

  public:

    /** Constructor */
    PlayerEvent();

    /** turn number */
    int turn;

  };
 }
}

#endif
