#include <iostream>

#include <qapplication.h>
#include <qlabel.h>
#include <qtable.h>
#include <qmessagebox.h>

#include "UAL/UI/Arguments.hh"

#include "UAL/QT/Player/RightAlignedTableItem.hh"
#include "UAL/QT/Player/TunefitEditor.hh"

UAL::QT::TunefitEditor::TunefitEditor(QWidget* parent, const char *name)
  : UAL::QT::BasicEditor(parent, name)
{
  isChanged = false;
  // Parent table
  initTable();
}

void UAL::QT::TunefitEditor::setShell(UAL::QT::PlayerShell* shell)
{
  p_shell = shell; // static_cast<UAL_RHIC::GtShell*>(shell);

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  table->setText(0, 1, QString::number(optics.m_chrom->twiss().mu(0)/(2.*UAL::pi)));
  table->setText(1, 1, QString::number(optics.m_chrom->twiss().mu(1)/(2.*UAL::pi)));
}

void UAL::QT::TunefitEditor::setPlayer(UAL::QT::BasicPlayer* player)
{
  p_player = player; // static_cast<UAL_RHIC::GtPlayer*>(player);
}


void UAL::QT::TunefitEditor::initTable()
{
  label->setText( tr( "Tunefit Parameters" ) );

  table->setNumCols(2);

  table->horizontalHeader()->setLabel(0, tr("Parameter"));
  table->setColumnReadOnly(0, true);
  table->setColumnWidth(0, 175);

  table->horizontalHeader()->setLabel(1, tr("Value"));
  table->setColumnReadOnly(1, false);
  table->setColumnWidth(1, 100);

  table->setColumnStretchable(0, false);
  table->setColumnStretchable(1, true);

  table->setNumRows(4);

  table->setText(0, 0, tr("hor. tune"));
  table->setText(0, 1, tr(""));

  table->setText(1, 0, tr("ver. tune"));
  table->setText(1, 1, tr(""));

  table->setText(2, 0, tr("hor. quadrupoles"));
  table->setItem(2, 1, new UAL::QT::RightAlignedTableItem(table, 
							  QTableItem::WhenCurrent, 
							  tr("")));

  table->setText(3, 0, tr("ver. quadrupoles"));
  table->setItem(3, 1, new UAL::QT::RightAlignedTableItem(table, 
							  QTableItem::WhenCurrent, 
							  tr("")));


}

void UAL::QT::TunefitEditor::setValue(int row, int col)
{
  isChanged = true;
}

void UAL::QT::TunefitEditor::updateData()
{
  bool ok;
  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  if(optics.m_chrom) {
    table->setText(0, 1, QString::number(optics.m_chrom->twiss().mu(0)/(2.*UAL::pi)));
    table->setText(1, 1, QString::number(optics.m_chrom->twiss().mu(1)/(2.*UAL::pi)));
  } else {
    table->setText(0, 1, QString::number(0.0));
    table->setText(1, 1, QString::number(0.0));
  }
}

void UAL::QT::TunefitEditor::activateChanges()
{

  bool ok;
  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  /*
  if(optics.m_chrom) {
    table->setText(0, 1, QString::number(optics.m_chrom->twiss().mu(0)/(2.*UAL::pi)));
    table->setText(1, 1, QString::number(optics.m_chrom->twiss().mu(1)/(2.*UAL::pi)));
  } else {
    table->setText(0, 1, QString::number(0.0));
    table->setText(1, 1, QString::number(0.0));
  }
  */

  if(!isChanged) return;

  double tunex       = table->text(0, 1).toDouble(&ok);
  double tuney       = table->text(1, 1).toDouble(&ok);
  std::string b1f     = table->text(2, 1);
  std::string b1d     = table->text(3, 1);

  if(b1f.length() == 0 || b1d.length() == 0) return;

  optics.tunefit(tunex, tuney, b1f, b1d);
  optics.calculate();

  table->setText(0, 1, QString::number(optics.m_chrom->twiss().mu(0)/(2.*UAL::pi)));
  table->setText(1, 1, QString::number(optics.m_chrom->twiss().mu(1)/(2.*UAL::pi)));

  isChanged = false;

  // p_shell->initRun();
  // p_player->update(0);
}




