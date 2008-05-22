#include <qapplication.h>

#include <Inventor/Qt/SoQt.h>
#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>

#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoText2.h>
#include <Inventor/nodes/SoLightModel.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoMaterialBinding.h>

#include <Inventor/nodes/SoPointSet.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoFont.h>
#include <Inventor/nodes/SoShuttle.h>
#include <Inventor/nodes/SoMaterial.h>

#include "UAL/COIN3D/Viewers/BunchViewer.hh"

int UAL::COIN3D::BunchViewer::counter = 0;

UAL::COIN3D::BunchViewer::BunchViewer(UAL::QT::BasicPlayer* player, 
				      PAC::Bunch* bunch)
  : UAL::QT::BasicViewer()
{

  p_player = player;
  p_bunch  = bunch;

  calculateAxisRanges();

  viewer = new SoQtExaminerViewer(m_frame);

  m_root = new SoSeparator;
  m_root->ref();

  setAxis();
   	
  // draw style
  SoDrawStyle * dstyle = new SoDrawStyle;
  dstyle->style.setValue( SoDrawStyle::POINTS );
  dstyle->pointSize.setValue( 2 );
  m_root->addChild( dstyle );

  // light model
  SoLightModel * lmodel = new SoLightModel;
  lmodel->model.setValue( SoLightModel::BASE_COLOR );
  m_root->addChild( lmodel );

  // color
  cube_col = new SoBaseColor;
  cube_col->rgb.setValue( 0, 0, 1 ); // red, green, blue
  m_root->addChild( cube_col );

  // material binding
  SoMaterialBinding * matbind = new SoMaterialBinding;
  matbind->value.setValue( SoMaterialBinding::PER_VERTEX );
  m_root->addChild( matbind );

  // create cube coordset
  p_point_coords = new SoCoordinate3;
  setPoints();
  m_root->addChild( p_point_coords );

  // cube pointset
  SoPointSet * pointset = new SoPointSet;
  m_root->addChild( pointset );

  viewer->setSceneGraph(m_root);

}

void UAL::COIN3D::BunchViewer::updateViewer(int turn)
{

  setPoints();
  cube_col->rgb.setValue( 0, 1, 0 );
  // p_point_coords->point.touch();
  viewer->render();
}

void UAL::COIN3D::BunchViewer::closeEvent(QCloseEvent* ce)
{

  ce->accept();

  m_root->unref();
  p_point_coords = 0;

  // p_page->closePlot();
  p_player->removeViewer("UAL::COIN3D::BunchViewer");

}


void UAL::COIN3D::BunchViewer::setPoints()
{
  if(!p_bunch) return;

  int size = p_bunch->size();

  double x0       = m_x0;
  double dx_max_0 = m_xmax - x0;  
  double dx_min_0 = x0 - m_xmin;  

  double y0       = m_y0;
  double dy_max_0 = m_ymax - y0;  
  double dy_min_0 = y0 - m_ymin; 

  double s0       = m_s0;
  double ds_max_0 = m_smax - s0;  
  double ds_min_0 = s0 - m_smin; 

  // Start update transaction
  p_point_coords->point.enableNotify(FALSE);

  p_point_coords->point.setNum(size);
  for(int i=0; i < size; i++) {
    PAC::Position& p = (*p_bunch)[i].getPosition();

    double x = p.getX();
    if(x > x0) { 
      x =  (x - x0)/dx_max_0; 
    } else { 
      x =  (x - x0)/dx_min_0;
    }

    double y = p.getY();
    if(y > y0) {
      y =  (y - y0)/dy_max_0;
    } else {
      y =  (y - y0)/dy_min_0;
    }

    double s = -p.getCT();
    if(s > s0) {
      s =  (s - s0)/ds_max_0;
    } else {
      s =  (s - s0)/ds_min_0;
    }

    p_point_coords->point.set1Value(i, x, y, s);
    // if(i == 0) std::cout << "s = " << s << std::endl;

  } 
  // End update transaction
  p_point_coords->point.enableNotify(TRUE);

  std::cout << "bunch 3d viewer: set points " << std::endl;

}

void UAL::COIN3D::BunchViewer::setAxis()
{

  double x0       = m_x0;
  double dx       = m_xmax - m_xmin;  

  double y0       = m_y0;
  double dy       = m_ymax - m_ymin;  

  double s0       = m_s0;
  double ds       = m_smax - m_smin;  

  SoFont * font = new SoFont;
  font->name.setValue("Courier");
  font->size.setValue(48.0);

  double factor = 1.2;

  // X
  SoCube* x_axis = new SoCube();
  x_axis->width  = factor*1.0;
  x_axis->height = 0.01;
  x_axis->depth =  0.01;

  SoShuttle * shuttlex = new SoShuttle;
  shuttlex->translation0.setValue(0.0, m_y0/dy, m_s0/ds);
  shuttlex->speed = 0;

  SoSeparator * x_axis_sep = new SoSeparator;
  x_axis_sep->addChild(shuttlex);
  x_axis_sep->addChild(x_axis);

  m_root->addChild(x_axis_sep);
	
  SoSeparator * sepx = new SoSeparator;
  sepx->addChild(font);
  sepx->addChild(text("X", SbVec3f(factor*m_xmax/dx, m_y0/dy, m_s0/ds)));

  m_root->addChild(sepx);

  // Y

  SoCube* y_axis = new SoCube();
  y_axis->width  = 0.01;
  y_axis->height = factor*1.00;
  y_axis->depth  = 0.01;

  SoShuttle * shuttley = new SoShuttle;
  shuttley->translation0.setValue(m_x0/dx, 0.0, m_s0/ds);
  shuttley->speed = 0;

  SoSeparator * y_axis_sep = new SoSeparator;
  y_axis_sep->addChild(shuttley);
  y_axis_sep->addChild(y_axis);

  m_root->addChild(y_axis_sep);

  SoSeparator * sepy = new SoSeparator;
  sepy->addChild(font);
  sepy->addChild(text("Y", SbVec3f(m_x0/dx, factor*m_ymax/dy, m_s0/ds)));
  m_root->addChild(sepy);

// Z

  SoCube* z_axis = new SoCube();
  z_axis->width  = 0.01;
  z_axis->height = 0.01;
  z_axis->depth  = 2.00;

  SoShuttle * shuttlez = new SoShuttle;
  shuttlez->translation0.setValue(m_x0/dx, m_y0/dy, 0.0);
  shuttlez->speed = 0;

  SoSeparator * z_axis_sep = new SoSeparator;
  z_axis_sep->addChild(shuttlez);
  z_axis_sep->addChild(z_axis);

  m_root->addChild(z_axis_sep);

  SoSeparator * sepz = new SoSeparator;
  sepz->addChild(font);
  sepz->addChild(text("Z", SbVec3f(m_x0/dx, m_y0/dy, 1.0)));

  m_root->addChild(sepz);

}

SoSeparator * UAL::COIN3D::BunchViewer::text(char * txtstring, SbVec3f pos1)
{
  SoSeparator * sep = new SoSeparator;
  SoShuttle * shuttle = new SoShuttle;
  SoText2 * txt = new SoText2;
  float x, y, z;

  txt->string.setValue(txtstring);
  pos1.getValue(x, y, z);
  shuttle->translation0.setValue(x, y, z);
  shuttle->speed = 0;

  sep->addChild(shuttle);
  sep->addChild(txt);

  return sep;
}

void UAL::COIN3D::BunchViewer::calculateAxisRanges()
{
  if(!p_bunch) {
    zeroAxisRanges();
    return;
  }

  m_xmax = m_ymax = m_smax = -1.e+23;
  m_xmin = m_ymin = m_smin =  1.e+23;

  int size = p_bunch->size();

  for(int i=0; i < size; i++) {
    PAC::Position& p = (*p_bunch)[i].getPosition();

    double x =  fabs(p.getX());
    if(x > m_xmax) m_xmax = x;
    // if(x < m_xmin) m_xmin = x;

    double y =  fabs(p.getY());
    if(y > m_ymax) m_ymax = y;
    // if(y < m_ymin) m_ymin = y;

    double s = fabs(-p.getCT());
    if(s > m_smax) m_smax = s;
    // if(s < m_smin) m_smin = s;
  }

  m_xmax *= 10;
  m_xmin  = -m_xmax;
  m_ymax *= 10;
  m_ymin  = -m_ymax;

  m_smin  = -m_smax;

  m_x0 = 0.0; // (m_xmax + m_xmin)/2.0;
  m_y0 = 0.0; // (m_ymax + m_ymin)/2.0;
  m_s0 = 0.0; // (m_smax + m_smin)/2.0;

}

void UAL::COIN3D::BunchViewer::zeroAxisRanges()
{
  m_xmin = m_x0 = m_xmax = 0.0;
  m_ymin = m_y0 = m_ymax = 0.0;
  m_smin = m_s0 = m_smax = 0.0;
}

