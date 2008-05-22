#ifndef UAL_QT_CHROMFIT_EDITOR_HH
#define UAL_QT_CHROMFIT_EDITOR_HH

#include <qlistview.h>

#include "UAL/QT/Player/BasicEditor.hh"
#include "UAL/QT/Player/PlayerShell.hh"
#include "UAL/QT/Player/BasicPlayer.hh"

namespace UAL
{
 namespace QT {

  class ChromfitEditor : public BasicEditor
  {

    Q_OBJECT

  public:

    /** Constructor */
    ChromfitEditor(QWidget* parent = 0, const char *name = 0);

    /** BasicEditor method called by player's setup button */
    void activateChanges();
    void updateData();

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

    bool isChanged;
    void initTable();

  };
 }
}

#endif
