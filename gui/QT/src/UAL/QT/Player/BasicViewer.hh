#ifndef UAL_GUI_BASIC_VIEWER_HH
#define UAL_GUI_BASIC_VIEWER_HH

#include <qmainwindow.h>
#include <qaction.h>
#include <qpopupmenu.h>
#include <qvbox.h>

namespace UAL
{
 namespace QT {
  class BasicViewer : public QMainWindow
  {

    Q_OBJECT

  public:

    /** Constructor */
    BasicViewer();

    virtual void updateViewer(int turn) { }
    virtual void setTurns(int turns) { m_turns = turns; }
    virtual void setFprint(int fprint) { m_fprint = fprint; }

  protected:

    virtual void saveFile(const QString& fileName);

  protected:

    void createActions();
    void createMenus();

  protected:

    QPopupMenu *fileMenu;
    QAction *saveAsAct;

    QString fileFilters;

  protected:

    QVBox* m_frame;

    int m_turns;
    int m_fprint;

  private slots:

    bool saveAs();

  };
 }
}

#endif
