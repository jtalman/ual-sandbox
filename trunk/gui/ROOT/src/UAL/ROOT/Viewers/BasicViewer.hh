#ifndef UAL_ROOT_BASIC_VIEWER_HH
#define UAL_ROOT_BASIC_VIEWER_HH

#include "TCanvas.h"
#include "TQtWidget.h"

#include "UAL/QT/Player/BasicViewer.hh"

namespace UAL
{
 namespace ROOT {

  class BasicViewer : public UAL::QT::BasicViewer
  {

    Q_OBJECT

  public:

    /** Constructor */
    BasicViewer();

  protected:

    virtual void saveFile(const QString& fileName);

  protected:

    // TQtCanvasWidget *viewer;
    TQtWidget *viewer; // ROOT embedded TCanvas

  public slots:
    virtual void processRootEvent(TObject *, unsigned int, TCanvas *);
  };
 }
}

#endif
