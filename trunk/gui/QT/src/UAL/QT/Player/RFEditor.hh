#ifndef UAL_QT_RF_EDITOR_HH
#define UAL_QT_RF_EDITOR_HH

#include <qlistview.h>

#include "UAL/QT/Player/BasicEditor.hh"
#include "UAL/QT/Player/PlayerShell.hh"
#include "UAL/QT/Player/BasicPlayer.hh"

namespace UAL
{
 namespace QT {

  class RFEditor : public BasicEditor
  {

    Q_OBJECT

  public:

    /** Constructor */
    RFEditor(QWidget* parent = 0, const char *name = 0);

    /** BasicEditor virtual method called by player's setup button */
    void activateChanges();

    /** Sets a player (application view) */
    void setPlayer(BasicPlayer* player);

    /** Sets a shell (application model) */
    void setShell(PlayerShell* shell);

   public slots:

    /** Sets value (called by table.valueChanged())  */
    void setValue(int row, int col);

  protected:

    /** Application View */
    BasicPlayer* p_player;

    /** Application Model */
    PlayerShell* p_shell;

  private:

    void initTable();
    void setOldValue(int row);

  };
 }
}

#endif
