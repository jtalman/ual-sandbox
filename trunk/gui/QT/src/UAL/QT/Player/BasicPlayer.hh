#ifndef UAL_QT_BASIC_PLAYER_HH
#define UAL_QT_BASIC_PLAYER_HH

#include <map>

#include <qwidget.h>
#include <qlistview.h>

#include "UAL/QT/Player/PlayerShell.hh"
#include "UAL/QT/Player/PlayerWorker.hh"
#include "UAL/QT/Player/MainPlayerUI.hh"
#include "UAL/QT/Player/BasicViewer.hh"
#include "UAL/QT/Player/BasicEditor.hh"

class QAction;

namespace UAL
{
  namespace QT {

    /** Main window of the Interactive Player */
    class BasicPlayer : public MainPlayerUI {

      Q_OBJECT

    public:

      /** Constructor */
      BasicPlayer(); // QWidget *parent = 0, const char *name = 0);

      /** Set a pointer to non-gui shell */
      void setShell(PlayerShell* shell);

      /** Returns a non-gui shell */
      PlayerShell* getShell() { return p_shell; }

      /** Showes page selected in the listView */
      virtual void showPage(QListViewItem* item);

      /** Adds a viewer */
      void addViewer(const std::string& className, BasicViewer* viewer);

      /** Removes a viewer */
      void removeViewer(const std::string& className);

      /** Adds an editor */
      void addEditor(const std::string& className, BasicEditor* editor);

      /** Removes an editor */
      void removeEditor(const std::string& className);

      /** Initializes application's parameters; 
       *  slot called by the "setup" button"
       */
      virtual void initRun();

      /** Starts a  thread (called by a "start button") */
    void startRun();
    
    void pauseRun();
    void continueRun();

    void stopRun();  

    /** Posts a custom event (called  by the model thread) */
    void update(int turn);

    /** Processes an event posted by the update method */
    void customEvent(QCustomEvent* customEvent);

    /** Sets a number of turns to be tracked */
    void setTurns(int turns)   { m_turns = turns; }

    /** Returns a number of turns */
    int  getTurns()            { return m_turns; }

    /** Sets a plotting frequency */
    void setFprint(int fprint) { m_fprint = fprint; }

    /** Returns a plotting frequency*/
    int  getFprint()           { return m_fprint; }

  protected:

    /** Non-gui shell */
    PlayerShell* p_shell;

    /** Model thread */
    PlayerWorker m_worker;

  protected:

    /** number of turns */
    int m_turns;

    /** plotting frequency */
    int m_fprint;

    /** Collection of viewers */
    std::map<std::string, BasicViewer*> m_viewers;

    /** Collection of editors */
    std::map<std::string, BasicEditor*> m_editors;

    protected:

      void createActions();
      void createMenus();  

    protected:

      void cleanSMF();
      void printStatistics(); 

    protected:

      // QPopupMenu *fileMenu;
      QPopupMenu *latticeMenu;
      QPopupMenu *propagatorMenu;

      QAction *openSxfAct; 
      QAction *saveSxfAct; 

      QAction *openApdfAct;  

  private slots:

    bool openSxf();
    bool saveSxf();
    bool openApdf();      
  };
 }
}

#endif
