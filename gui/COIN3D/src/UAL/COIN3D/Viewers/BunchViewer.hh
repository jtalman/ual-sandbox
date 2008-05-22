#ifndef UAL_COIN3D_BUNCH_VIEWER_HH
#define UAL_COUN3D_BUNCH_VIEWER_HH

#include <qvbox.h>
#include <qlistview.h>
#include <qevent.h>

#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoBaseColor.h>

#include "PAC/Beam/Bunch.hh"

#include "UAL/QT/Player/BasicPlayer.hh"
#include "UAL/QT/Player/BasicViewer.hh"

namespace UAL
{
 namespace COIN3D { 

  class BunchViewer : public UAL::QT::BasicViewer
  {

    Q_OBJECT

  public:

    /** Constructor */
    BunchViewer(UAL::QT::BasicPlayer* page, PAC::Bunch* bunch);

    void updateViewer(int turn);

    /** Closes window and clean up state */
    void closeEvent(QCloseEvent* ce);

    /** Updates points */
    void setPoints();

  private:

    void zeroAxisRanges();
    void calculateAxisRanges();

    double m_xmin, m_x0, m_xmax;
    double m_ymin, m_y0, m_ymax;
    double m_smin, m_s0, m_smax;

    PAC::Bunch* p_bunch;

  private:

    static int counter;

    // Parent player
    UAL::QT::BasicPlayer* p_player;

    SoQtExaminerViewer* viewer;

    // root node of the Coin examiner

    SoSeparator* m_root;

    // pointer to the container with points
    SoCoordinate3* p_point_coords;

    SoBaseColor * cube_col;

  private:

    void setAxis();
    SoSeparator* text(char* txtstring, SbVec3f pos1);

  };

 };

};

#endif
