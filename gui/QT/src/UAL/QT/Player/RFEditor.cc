#include <iostream>

#include <qapplication.h>
#include <qlabel.h>
#include <qtable.h>
#include <qmessagebox.h>

#include "UAL/UI/Arguments.hh"

#include "UAL/QT/Player/RFEditor.hh"
#include "TIBETAN/Propagator/RFCavityTracker.hh"

UAL::QT::RFEditor::RFEditor(QWidget* parent, const char *name)
  : UAL::QT::BasicEditor(parent, name)
{
  // Parent table
  initTable();
}

void UAL::QT::RFEditor::initTable()
{
  label->setText( tr( "RF Parameters" ) );

  table->setNumCols(3);

  table->horizontalHeader()->setLabel(0, tr("Parameter"));
  table->setColumnReadOnly(0, true);
  table->setColumnWidth(0, 175);

  table->horizontalHeader()->setLabel(1, tr("Value"));
  table->setColumnReadOnly(1, false);
  table->setColumnWidth(1, 100);

  table->horizontalHeader()->setLabel(2, tr("Unit"));
  table->setColumnReadOnly(2, true);

  table->setColumnStretchable(0, false);
  table->setColumnStretchable(1, false);
  table->setColumnStretchable(2, true);

  table->setNumRows(6);

  table->setText(0, 0, tr("peak voltage:"));
  table->setText(0, 1, tr(""));
  table->setText(0, 2, tr("GV"));

  table->setText(1, 0, tr("phase:"));
  table->setText(1, 1, tr(""));
  table->setText(1, 2, tr("degrees"));

  table->setText(2, 0, tr("harmonic number:"));
  table->setText(2, 1, tr(""));
  table->setText(2, 2, tr(""));

}

void UAL::QT::RFEditor::setShell(UAL::QT::PlayerShell* shell)
{
  p_shell = shell;

  TIBETAN::RFCavityTracker& rf = p_shell->getRF();

  table->setText(0, 1, QString::number(rf.getV()));
  table->setText(1, 1, QString::number(360*rf.getLag()));
  table->setText(2, 1, QString::number(rf.getHarmon()));
}

void UAL::QT::RFEditor::setPlayer(UAL::QT::BasicPlayer* player)
{
  p_player = player;
}

void UAL::QT::RFEditor::setValue(int row, int col)
{
}

void UAL::QT::RFEditor::activateChanges()
{
  bool ok;

  double volt       = table->text(0, 1).toDouble(&ok);
  double phase      = table->text(1, 1).toDouble(&ok);
  double harmon     = table->text(2, 1).toDouble(&ok);

  p_shell->setRF(UAL::Args() 
		 << UAL::Arg("volt",   volt)
		 << UAL::Arg("lag",    phase/360.0)
		 << UAL::Arg("harmon", harmon));
}


void UAL::QT::RFEditor::setOldValue(int row)
{
  TIBETAN::RFCavityTracker& rf = p_shell->getRF();

  double oldValue = 0;
  switch(row) {
  case 0: 
    oldValue = rf.getV();
    break;
  case 1: 
    oldValue = 360*rf.getLag();
    break;  
  case 2: 
    oldValue = rf.getHarmon();
    break;  
  default:
    break;
  };

  table->setText(row, 1, QString::number(oldValue));
  table->update();
}



