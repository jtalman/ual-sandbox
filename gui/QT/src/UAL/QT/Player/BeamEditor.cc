#include <iostream>

#include <qapplication.h>
#include <qlabel.h>
#include <qtable.h>
#include <qmessagebox.h>

#include "UAL/UI/Arguments.hh"

#include "UAL/QT/Player/BeamEditor.hh"

UAL::QT::BeamEditor::BeamEditor(QWidget* parent, const char *name)
  : UAL::QT::BasicEditor(parent, name)
{
  // Parent table
  initTable();
}

void UAL::QT::BeamEditor::setShell(UAL::QT::PlayerShell* shell)
{
  p_shell = shell; // static_cast<UAL_RHIC::GtShell*>(shell);

  UAL::BunchGenerator& bg = p_shell->getBunchGenerator();

  table->setText(0, 1, bg.getType());
  table->setText(1, 1, QString::number(bg.getBunchSize()));

  PAC::BeamAttributes& ba = p_shell->getBeamAttributes();

  table->setText(2, 1, QString::number(ba.getEnergy()/ba.getMass()));
  table->setText(3, 1, QString::number(ba.getMass()));
  table->setText(4, 1, QString::number(ba.getCharge()));

  table->setText(5, 1, QString::number(bg.getEnX()));
  table->setText(6, 1, QString::number(bg.getEnY()));

  table->setText(7, 1, QString::number(p_shell->getCtHalfWidth()));
  table->setText(8, 1, QString::number(p_shell->getDeHalfWidth()));
}

void UAL::QT::BeamEditor::setPlayer(UAL::QT::BasicPlayer* player)
{
  p_player = player; // static_cast<UAL_RHIC::GtPlayer*>(player);
}


void UAL::QT::BeamEditor::initTable()
{
  label->setText( tr( "Beam Parameters" ) );

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

  table->setNumRows(9);

  table->setText(0, 0, tr("type:"));
  table->setText(0, 1, tr(""));
  table->setText(0, 2, tr(""));

  table->setText(1, 0, tr("particles per bunch:"));
  table->setText(1, 1, tr(""));
  table->setText(1, 2, tr(""));

  table->setText(2, 0, tr("gamma:"));
  table->setText(2, 1, tr(""));
  table->setText(2, 2, tr(""));

  table->setText(3, 0, tr("mass:"));
  table->setText(3, 1, tr(""));
  table->setText(3, 2, tr("GeV/c2"));

  table->setText(4, 0, tr("charge:"));
  table->setText(4, 1, tr(""));
  table->setText(4, 2, tr(""));

  table->setText(5, 0, tr("enx:"));
  table->setText(5, 1, tr(""));
  table->setText(5, 2, tr("pi*m*rad"));

  table->setText(6, 0, tr("eny:"));
  table->setText(6, 1, tr(""));
  table->setText(6, 2, tr("pi*m*rad"));

  table->setText(7, 0, tr("half bunch length, ct max:"));
  table->setText(7, 1, tr(""));
  table->setText(7, 2, tr("m"));

  table->setText(8, 0, tr("half energy spread, de/p0c:"));
  table->setText(8, 1, tr(""));
  table->setText(8, 2, tr(""));

}

void UAL::QT::BeamEditor::setValue(int row, int col)
{
  /*
  bool ok;
  double newValue = table->text(row, col).toDouble(&ok);

  if(!ok || newValue <= 0){
    QMessageBox::critical(0, tr("Beam Editor"),
			  tr("New entry is not a positive number"));
    setOldValue(row);
    return;
  }
  */

}

void UAL::QT::BeamEditor::updateData()
{
}

void UAL::QT::BeamEditor::activateChanges()
{
  bool ok;

  UAL::BunchGenerator& bg = p_shell->getBunchGenerator();

  std::string type   = table->text(0, 1).ascii();
  bg.setType(type);

  int npart          = (int) table->text(1, 1).toDouble(&ok);
  p_shell->setBunch(UAL::Args() << UAL::Arg("np", npart));

  double gamma       = table->text(2, 1).toDouble(&ok);
  double mass        = table->text(3, 1).toDouble(&ok);
  double charge      = table->text(4, 1).toDouble(&ok);

  p_shell->setBeamAttributes(UAL::Args() 
			  << UAL::Arg("charge", charge)
  			  << UAL::Arg("mass",   mass)
  			  << UAL::Arg("energy", gamma*mass));

  double enx = table->text(5, 1).toDouble(&ok);
  bg.setEnX(enx);

  double eny = table->text(6, 1).toDouble(&ok);
  bg.setEnY(eny);

  double ctHalfWidth = table->text(7, 1).toDouble(&ok);
  double deHalfWidth = table->text(8, 1).toDouble(&ok);

  p_shell->setLongEmittance(ctHalfWidth, deHalfWidth);

  // p_shell->initRun();
  // p_player->update(0);
}


void UAL::QT::BeamEditor::setOldValue(int row)
{
  double oldValue = 0;
  switch(row) {
  case 0: 
    oldValue = p_shell->getBunch().size();
    break;
  case 1: 
    oldValue = p_shell->getBeamAttributes().getEnergy()/p_shell->getBeamAttributes().getMass();
    break;  
  case 2: 
    oldValue = p_shell->getBeamAttributes().getMass();
    break;  
  case 3: 
    oldValue = p_shell->getBeamAttributes().getCharge();
    break;    
  case 4: 
    oldValue = p_shell->getCtHalfWidth();
    break;
  case 5: 
    oldValue = p_shell->getDeHalfWidth();
    break;
  default:
    break;
  };

  table->setText(row, 1, QString::number(oldValue));
  table->update();
}



