#ifndef UAL_QT_BASIC_EDITOR_HH
#define UAL_QT_BASIC_EDITOR_HH

#include "UAL/QT/Player/TablePageUI.hh"

namespace UAL
{
 namespace QT { 
  class BasicEditor : public  TablePageUI
  {

    Q_OBJECT

  public:

    /** Constructor */
    BasicEditor(QWidget* parent = 0, const char *name = 0);

    /** Destructor */
    virtual ~BasicEditor();

    virtual void activateChanges();
    virtual void updateData() {}

  };
 }
}

#endif
