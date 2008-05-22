

#include "UAL/QT/Player/RightAlignedTableItem.hh"

UAL::QT::RightAlignedTableItem::RightAlignedTableItem(QTable* table, 
						      EditType et, 
						      const QString& text)
  : QTableItem(table, et, text)
{
}
