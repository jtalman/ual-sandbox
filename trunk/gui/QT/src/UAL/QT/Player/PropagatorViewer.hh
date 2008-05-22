#ifndef UAL_QT_PROPAGATOR_VIEWER_HH
#define UAL_QT_PROPAGATOR_VIEWER_HH

#include <qlistview.h>

#include "UAL/QT/Player/BasicEditor.hh"
#include "UAL/QT/Player/PlayerShell.hh"
#include "UAL/QT/Player/BasicPlayer.hh"

namespace UAL
{
 namespace QT {

  class PropagatorViewer : public BasicEditor
  {

    Q_OBJECT

  public:

    /** Constructor */
   PropagatorViewer(QWidget* parent = 0, const char *name = 0);

    /** BasicEditor method called by player's setup button */
    void activateChanges();

   public slots:

    /** Sets value (called by table.valueChanged())  */
    void setValue(int row, int col);

  protected:

  private:

    void initTable();

  };
 }
}

#endif
