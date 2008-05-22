#include <iostream>

#include <qapplication.h>
#include <qlabel.h>
#include <qtable.h>
#include <qmessagebox.h>

#include "UAL/UI/Arguments.hh"
// #include "UAL/ADXF/ConstantManager.hh"

#include "UAL/QT/Player/ParameterViewer.hh"

UAL::QT::ParameterViewer::ParameterViewer(QWidget* parent, const char *name)
  : UAL::QT::BasicEditor(parent, name)
{
  // Parent table
  initTable();
}

void UAL::QT::ParameterViewer::initTable()
{
  label->setText( tr( "ADXF Constants" ) );

  table->setNumCols(2);

  table->horizontalHeader()->setLabel(0, tr("Constant"));
  table->setColumnReadOnly(0, true);
  table->setColumnWidth(0, 175);

  table->horizontalHeader()->setLabel(1, tr("Value"));
  table->setColumnReadOnly(1, true);
  table->setColumnWidth(1, 100);

  table->setColumnStretchable(0, false);
  table->setColumnStretchable(1, false);

  /*

  mu::Parser& muParser = UAL::ADXFConstantManager::getInstance()->muParser;

  const mu::Parser::valmap_type constants = muParser.GetConst();

  // cout << "\nParser constants:\n";
  // cout <<   "-----------------\n";
  // cout << "Number: " << (int) constants.size() << "\n";

  table->setNumRows(constants.size());

  int counter = 0;
  mu::Parser::valmap_type::const_iterator item = constants.begin();
  for (; item!=constants.end(); ++item){
    QString qstr(item->first);
    table->setText(counter, 0, tr(qstr));
    table->setText(counter, 1, QString::number(item->second));
    // cout << counter << ", Name: " << item->first << ", Value: " << item->second << "\n";
    counter++;
  }

  */

}

void UAL::QT::ParameterViewer::setValue(int row, int col)
{
}

void UAL::QT::ParameterViewer::activateChanges()
{
}






