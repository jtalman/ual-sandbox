#include <iostream>

#include <qaction.h>
#include <qapplication.h>
#include <qcombobox.h>
#include <qfiledialog.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qmenubar.h>
#include <qmessagebox.h>
#include <qpopupmenu.h>
#include <qsettings.h>
#include <qstatusbar.h>
#include <qstringlist.h>
#include <qlistbox.h>
#include <qlistview.h>
#include <qslider.h>
#include <qwidgetstack.h>
#include <qpushbutton.h>

#include "UAL/UI/OpticsCalculator.hh"

#include "SMF/PacSmf.h"
#include "UAL/SMF/AcceleratorNodeFinder.hh"

#include "UAL/QT/Player/PlayerEvent.hh"
#include "UAL/QT/Player/BasicPlayer.hh"

// UAL::QT::BasicPlayer::BasicPlayer(QWidget *parent, const char *name)
//   : PlayerUI(parent, name)
UAL::QT::BasicPlayer::BasicPlayer()
  : MainPlayerUI()
{

  //  m_frame = new PlayerUI(this);
  // setCentralWidget(m_frame);

  int w = 500;
  int h = 300;
  int x = QApplication::desktop()->width()/2 - w/2;
  int y = QApplication::desktop()->height()/2 - h/2; 

  setGeometry(x, y, w, h); 

  p_shell    = 0;

  m_turns    =  0;
  m_fprint   = -1;

  m_worker.setPlayer(this);

  setupButton->setEnabled(true);
  runButton->setEnabled(false);
  pauseButton->setEnabled(false);
  continueButton->setEnabled(false);
  stopButton->setEnabled(false);

  createActions();
  createMenus();

}

void UAL::QT::BasicPlayer::setShell(UAL::QT::PlayerShell* shell)
{
  p_shell = shell;
  p_shell->initRun();

  m_worker.setShell(p_shell);

}

void UAL::QT::BasicPlayer::createActions()
{
  openApdfAct = new QAction(tr("Open &APDF..."), tr("Open &APDF..."), 0, this);
  connect(openApdfAct, SIGNAL(activated()), this, SLOT(openApdf()));

  openSxfAct = new QAction(tr("Open SXF..."), tr("Open SXF..."), 0, this);
  connect(openSxfAct, SIGNAL(activated()), this, SLOT(openSxf()));

  saveSxfAct = new QAction(tr("Save SXF..."), tr("Save SXF..."), 0, this);
  connect(saveSxfAct, SIGNAL(activated()), this, SLOT(saveSxf()));
}

void UAL::QT::BasicPlayer::createMenus()
{
  latticeMenu = new QPopupMenu(this);
  openSxfAct->addTo(latticeMenu);
  saveSxfAct->addTo(latticeMenu); 
  menuBar()->insertItem(tr("&Lattice"), latticeMenu);


  propagatorMenu = new QPopupMenu(this);
  openApdfAct->addTo(propagatorMenu);
  menuBar()->insertItem(tr("&Propagator"), propagatorMenu);
}


void UAL::QT::BasicPlayer::printStatistics()
{

  PacSmf smf;
  PacLattices* lattices = smf.lattices();
  std::cout << "lattice " << lattices->size() << std::endl;

  PacLines* lines = smf.lines();
  std::cout << "lines " << lines->size() << std::endl;

  PacGenElements* genElements = smf.elements();
  std::cout << "element " << genElements->size() << std::endl;

  UAL::AcceleratorNodeFinder& nf =  UAL::AcceleratorNodeFinder::getInstance();
  std::cout << "nodes " << nf.size() << std::endl;;

}

void UAL::QT::BasicPlayer::cleanSMF()
{

  PacSmf smf;
  PacLattices* lattices = smf.lattices();
  lattices->clean();

  PacLines* lines = smf.lines();
  lines->clean();

  PacGenElements* genElements = smf.elements();
  genElements->clean();


  UAL::AcceleratorNodeFinder& nf =  UAL::AcceleratorNodeFinder::getInstance();
  nf.clean();

}

bool UAL::QT::BasicPlayer::openSxf()
{
  QString sxfFilter = tr("SXF files (*.sxf)");

  QString fileName = QFileDialog::getOpenFileName(".", sxfFilter, this);
  if (fileName.isEmpty())
    return false;
    
  if (!fileName.isEmpty()) {
    // printStatistics();
    cleanSMF();
    // printStatistics();
    p_shell->readSXF(Args() << Arg("file", fileName.ascii() )); 
    p_shell->use(UAL::Args() << UAL::Arg("lattice", "ring"));
    // printStatistics();
  }
  return true;
}

bool UAL::QT::BasicPlayer::saveSxf()
{
  QString fileName = QFileDialog::getSaveFileName(".", QString::null, this);
  if (fileName.isEmpty())
    return false;
    
  if (!fileName.isEmpty()) {
    p_shell->writeSXF(Args() << Arg("file", fileName.ascii() )); 
  }
  return true;
}

bool UAL::QT::BasicPlayer::openApdf()
{

  QString apdfFilter = tr("ADXF files (*.apdf)");
  QString fileName = QFileDialog::getOpenFileName(".", apdfFilter, this);
  if (fileName.isEmpty())
    return false;
    
  if (!fileName.isEmpty()) {
    p_shell->readAPDF(Args() << Arg("file", fileName.ascii() )); 
  }
  return true;
}


void UAL::QT::BasicPlayer::showPage(QListViewItem* item)
{
}

void UAL::QT::BasicPlayer::addEditor(const std::string& className, UAL::QT::BasicEditor* editor)
{
  m_editors[className] = editor;
}

void UAL::QT::BasicPlayer::removeEditor(const std::string& className)
{
  m_editors.erase(className);
}


void UAL::QT::BasicPlayer::addViewer(const std::string& className, UAL::QT::BasicViewer* viewer)
{
  m_viewers[className] = viewer;
}

void UAL::QT::BasicPlayer::removeViewer(const std::string& className)
{
  m_viewers.erase(className);
}

void UAL::QT::BasicPlayer::initRun()
{
  turnSlider->setRange(0, m_turns);

  std::map<std::string, UAL::QT::BasicEditor* >::iterator it;
  for(it = m_editors.begin(); it != m_editors.end(); it++){
    it->second->activateChanges();
  }

  setupButton->setEnabled(true);
  runButton->setEnabled(true);
  pauseButton->setEnabled(false);
  continueButton->setEnabled(false);
  stopButton->setEnabled(false);

  m_worker.initRun();

  for(it = m_editors.begin(); it != m_editors.end(); it++){
    it->second->updateData();
  }

}

void UAL::QT::BasicPlayer::startRun()
{
  turnSlider->setRange(0, m_turns);

  setupButton->setEnabled(false);
  runButton->setEnabled(false);
  pauseButton->setEnabled(true);
  continueButton->setEnabled(false);
  stopButton->setEnabled(true);

  m_worker.startRun();
}

void UAL::QT::BasicPlayer::pauseRun()
{
  m_worker.pauseRun();

  setupButton->setEnabled(false);
  runButton->setEnabled(false);
  pauseButton->setEnabled(false);
  continueButton->setEnabled(true);
  stopButton->setEnabled(false);

}

void UAL::QT::BasicPlayer::continueRun()
{
  setupButton->setEnabled(false);
  runButton->setEnabled(false);
  pauseButton->setEnabled(true);
  continueButton->setEnabled(false);
  stopButton->setEnabled(true);

  m_worker.continueRun();

}

void UAL::QT::BasicPlayer::stopRun()
{
  m_worker.stopRun();

  setupButton->setEnabled(true);
  runButton->setEnabled(false);
  pauseButton->setEnabled(false);
  continueButton->setEnabled(false);
  stopButton->setEnabled(false);
}

void UAL::QT::BasicPlayer::update(int turn)
{

  int t = m_turns/100;
  if(((turn/t)*t ) == turn) {
    turnSlider->setValue(turn);
  }

  if((turn/m_fprint)*m_fprint == turn) {
    // std::cout << "basic player: post event " << turn << std::endl;
    UAL::QT::PlayerEvent* event = new UAL::QT::PlayerEvent();
    event->turn = turn;
    QApplication::postEvent(this, event);
    std::cout << "UAL::QT::BasicPlayer::update: update and post event " << turn << std::endl;
  }
}

void UAL::QT::BasicPlayer::customEvent(QCustomEvent* customEvent)
{
  if((int) customEvent->type() != 65432) return;

  UAL::QT::PlayerEvent* playerEvent = (UAL::QT::PlayerEvent*) customEvent;
  int turn = playerEvent->turn;

  m_worker.pauseRun();

  std::map<std::string, UAL::QT::BasicViewer* >::iterator it;
  for(it = m_viewers.begin(); it != m_viewers.end(); it++){
    it->second->updateViewer(turn);
  }

  m_worker.continueRun();
}


