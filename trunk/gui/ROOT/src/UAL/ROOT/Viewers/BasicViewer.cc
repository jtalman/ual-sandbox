#include <iostream>

#include <qapplication.h>
#include <qmenubar.h>
#include <qfiledialog.h>
#include <qmessagebox.h>
#include <qtooltip.h>
#include <qstatusbar.h>
#include <qtextstream.h>

#include "TCanvas.h"

#include "UAL/ROOT/Viewers/BasicViewer.hh"

UAL::ROOT::BasicViewer::BasicViewer()
  : UAL::QT::BasicViewer()
{
  viewer = new TQtWidget(m_frame,"EmbeddedCanvas");
  connect(viewer,SIGNAL(RootEventProcessed(TObject *, unsigned int, TCanvas *)),
          this,SLOT(processRootEvent(TObject *, unsigned int, TCanvas *)));
  viewer->EnableSignalEvents(kMousePressEvent);

  /*
  QToolTipGroup * m_tipGroup = new QToolTipGroup( this );
  connect(m_tipGroup, SIGNAL(showTip(const QString&)), statusBar(),
           SLOT(message(const QString&)) );
  connect(m_tipGroup, SIGNAL(removeTip()), statusBar(), SLOT(clear()) );
  */
}

void  UAL::ROOT::BasicViewer::processRootEvent(TObject *obj, unsigned int, TCanvas *)
{
  // std::cout << "processRootEvent " << obj->ClassName() << " " << obj->GetName() << std::endl;

  TQtWidget *tipped = (TQtWidget *)sender();
  /*
  const char *objectInfo =
        obj->GetObjectInfo(tipped->GetEventX(),tipped->GetEventY());
  QString tipText; //  ="You have ";
  */

  /*
  if  (tipped == tQtWidget1)
     tipText +="clicked";
  else
     tipText +="passed";
  */

  /*
  std::cout << "tipped x=" << tipped->GetEventX() << "y= " << tipped->GetEventY() << std::endl;
  std::cout << "canvas x=" << viewer->GetCanvas()->PadtoX(tipped->GetEventX())
	    << "y= " << viewer->GetCanvas()->PadtoY(tipped->GetEventY()) << std::endl; ;
  std::cout << "pad x=" << viewer->GetSelectedPad()->PadtoX(tipped->GetEventX())
	    << "y= " << viewer->GetSelectedPad()->PadtoY(tipped->GetEventY()) << std::endl;
  std::cout << "pad abs x=" << viewer->GetSelectedPad()->AbsPixeltoX(tipped->GetEventX())
	    << "y= " << viewer->GetSelectedPad()->AbsPixeltoY(tipped->GetEventY()) << std::endl;
  */
 
  /*
  tipText += " the object <";
  tipText += obj->GetName();
  tipText += "> of class ";
  tipText += obj->ClassName();
  tipText += " : ";

  tipText += objectInfo;

  QToolTip::setWakeUpDelay(1);
  QToolTip::remove(tipped);
  QToolTip::add(tipped,QRect(tipped->GetEventX()-5,tipped->GetEventY()-5,10,10),tipText);
  */

  QString str;
  QTextOStream(&str) << "x=" << viewer->GetSelectedPad()->AbsPixeltoX(tipped->GetEventX()) 
		    <<",y=" << viewer->GetSelectedPad()->AbsPixeltoY(tipped->GetEventY());

  statusBar()->message(str);

}


void UAL::ROOT::BasicViewer::saveFile(const QString& fileName)
{
  // std::cout << "fileName = " << fileName << std::endl;
  // viewer->GetCanvas()->SaveAs(fileName);
  QPixmap screenCapture = QPixmap::grabWidget(m_frame);
  screenCapture.save(fileName,"PNG");
}
