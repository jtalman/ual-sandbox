#include <qapplication.h>
#include <qmenubar.h>
#include <qfiledialog.h>
#include <qmessagebox.h>

#include "UAL/QT/Player/BasicViewer.hh"

UAL::QT::BasicViewer::BasicViewer()
  : QMainWindow()
{
  m_frame = new QVBox(this);
  setCentralWidget(m_frame);

  m_frame->setMargin(10);
  m_frame->setFrameStyle( QFrame::StyledPanel | QFrame::Sunken );

  int w = 400;
  int h = 300;
  int x = QApplication::desktop()->width()/2 - w/2;
  int y = QApplication::desktop()->height()/2 - h/2;

  setGeometry(x, y, w, h);

  createActions();
  createMenus();

  /*
  fileFilters = tr("Postscript (*.ps)\n"
		   "Encapsulated Postscript (*.eps)\n"
		   "Image (*.bmp,*.jpeg,*.gif,*.pbm,*.pgm,*.png,*.ppm,*.xbm,*xpm)\n"
		   "All files (*.*)");
  */
  fileFilters = tr("Image (*.png)");
}

void UAL::QT::BasicViewer::createActions()
{
  saveAsAct = new QAction(tr("Save &As..."), tr("Save &As..."), 0, this);
  connect(saveAsAct, SIGNAL(activated()), this, SLOT(saveAs()));
}

void UAL::QT::BasicViewer::createMenus()
{
  fileMenu = new QPopupMenu(this);
  saveAsAct->addTo(fileMenu);

  menuBar()->insertItem(tr("&File"), fileMenu);
}

bool UAL::QT::BasicViewer::saveAs()
{
  QString fileName = QFileDialog::getSaveFileName(".", fileFilters, this);
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
    
  if (!fileName.isEmpty()) saveFile(fileName);
  return true;
}

void UAL::QT::BasicViewer::saveFile(const QString& fileName)
{

}



