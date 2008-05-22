#ifndef UAL_QT_RIGHT_ALIGNED_TABLE_ITEM_HH
#define UAL_QT_RIGHT_ALIGNED_TABLE_ITEM_HH

#include <iostream>
#include <qtable.h>

namespace UAL
{
 namespace QT { 
  class RightAlignedTableItem : public  QTableItem
  {

  public:

    /** Constructor */
    RightAlignedTableItem(QTable* table, EditType et, const QString& text);

    virtual int alignment() const { return Qt::AlignRight; }

  };
 }
}

#endif
