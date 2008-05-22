#ifndef UAL_QT_TURN_COUNTER_HH
#define UAL_QT_TURN_COUNTER_HH

namespace UAL
{
  namespace QT {

    class TurnCounter
    {
    public:

      /** Returns singleton */
      static TurnCounter* getInstance();

      void setTurn(int turn) { m_turn = turn; }

      int getTurn() { return m_turn; }

    protected:

      int m_turn;

    private:

      static TurnCounter* s_theInstance;

      TurnCounter();

    };
  }
}

#endif
