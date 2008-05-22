#include <iostream>

#include <qapplication.h>
#include <qlabel.h>
#include <qtable.h>
#include <qmessagebox.h>

#include "UAL/UI/Arguments.hh"
#include "UAL/APF/PropagatorFactory.hh"

#include "UAL/QT/Player/PropagatorViewer.hh"

UAL::QT::PropagatorViewer::PropagatorViewer(QWidget* parent, const char *name)
  : UAL::QT::BasicEditor(parent, name)
{
  // Parent table
  initTable();
}

void UAL::QT::PropagatorViewer::initTable()
{
  label->setText( tr( "Collection of Propagators" ) );

  table->setNumCols(2);

  table->horizontalHeader()->setLabel(0, tr("Propagator"));
  table->setColumnReadOnly(0, true);
  table->setColumnWidth(0, 175);

  table->horizontalHeader()->setLabel(1, tr("Description"));
  table->setColumnReadOnly(1, true);
  table->setColumnWidth(1, 100);

  table->setColumnStretchable(0, false);
  table->setColumnStretchable(1, false);

  UAL::PropagatorFactory& propagators = UAL::PropagatorFactory::getInstance();

  int counter = 0;
  UAL::PropagatorFactory::Iterator it;
  for(it = propagators.begin(); it != propagators.end(); it++){
    counter++;
  }

  table->setNumRows(counter);

  counter = 0;
  for (it = propagators.begin(); it != propagators.end(); it++){
    QString qstr(it->first);
    table->setText(counter++, 0, tr(qstr));
  }

}

void UAL::QT::PropagatorViewer::setValue(int row, int col)
{
}

void UAL::QT::PropagatorViewer::activateChanges()
{
}






