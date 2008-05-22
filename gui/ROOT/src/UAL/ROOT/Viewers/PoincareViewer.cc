
#include <qapplication.h>
#include <qmenubar.h>
#include <qfiledialog.h>
#include <qmessagebox.h>

#include "UAL/Common/Def.hh"

#include "AIM/BPM/PoincareMonitor.hh"
#include "AIM/BPM/PoincareMonitorCollector.hh"

#include "UAL/QT/Player/TurnCounter.hh"
#include "UAL/ROOT/Viewers/PoincareViewer.hh"

UAL::ROOT::PoincareViewer::PoincareViewer(UAL::QT::BasicPlayer* player)
  : UAL::ROOT::BasicViewer()
{

  updateMenu();

  p_player     = player;

  m_prevTurn   =  1000000000;

  m_bpmIndex   = 0;
  m_points     = p_player->getTurns();

  m_tinyGraph  = 0;


}

void UAL::ROOT::PoincareViewer::updateMenu()
{
  writeToAct = new QAction(tr("Write &To..."), tr("Write &To..."), 0, this);
  writeToAct->addTo(fileMenu);
  connect(writeToAct, SIGNAL(activated()), this, SLOT(writeTo()));
}

bool UAL::ROOT::PoincareViewer::writeTo()
{
  QString fileName = QFileDialog::getSaveFileName(".", tr("All files (*.*)"), this);
  if (fileName.isEmpty())
    return false;

  if (QFile::exists(fileName)) {
    int ret = QMessageBox::warning(this, tr("Warning"),
	   tr("File %1 already exists.\n"
	      "Do you want to overwrite it?")
               .arg(QDir::convertSeparators(fileName)),
	       QMessageBox::Yes | QMessageBox::Default,
               QMessageBox::No | QMessageBox::Escape);
    if (ret == QMessageBox::No) return true;
  }
    
  if (!fileName.isEmpty()) writeToFile(fileName);
  return true;
}

void UAL::ROOT::PoincareViewer::writeToFile(const QString& fileName)
{
  std::map<int, AIM::PoincareMonitor*>::iterator ibpms = 
    AIM::PoincareMonitorCollector::getInstance().getAllData().begin(); // find(m_bpmIndex);

  if(ibpms == AIM::PoincareMonitorCollector::getInstance().getAllData().end()) 
    return;

  AIM::PoincareMonitor* bpm = ibpms->second;

  bpm->write(fileName.ascii());
}


void UAL::ROOT::PoincareViewer::updateViewer(int turn)
{

}

void UAL::ROOT::PoincareViewer::closeEvent(QCloseEvent* ce)
{

}


