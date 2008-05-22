#ifndef UAL_QT_RF_EDITOR_HH
#define UAL_QT_RF_EDITOR_HH

#include <qlistview.h>

#include "UAL/QT/Player/BasicEditor.hh"

#include "GtShell.hh"
#include "GtPlayer.hh"

namespace UAL
{
 namespace USPAS {

  class RFEditor : public UAL::QT::BasicEditor
  {

    Q_OBJECT

  public:

    /** Constructor */
    RFEditor(QWidget* parent = 0, const char *name = 0);

    /** BasicEditor virtual method called by player's setup button */
    void activateChanges();

    /** Sets a player (application view) */
    void setPlayer(UAL::QT::BasicPlayer* player);

    /** Sets a shell (application model) */
    void setShell(UAL::QT::PlayerShell* shell);

   public slots:

    /** Sets value (called by table.valueChanged())  */
    void setValue(int row, int col);

  protected:

    /** Application View */
    GtPlayer* p_player;

    /** Application Model */
    GtShell* p_shell;

  private:

    void initTable();
    void setOldValue(int row);

  };
 }
}

#endif
