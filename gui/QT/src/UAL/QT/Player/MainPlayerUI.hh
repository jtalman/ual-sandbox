/****************************************************************************
** Form interface generated from reading ui file 'MainPlayerUI.ui'
**
** Created: Thu May 26 21:02:26 2005
**      by: The User Interface Compiler ($Id: MainPlayerUI.hh,v 1.1 2005/05/31 17:11:42 ual Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef MAINPLAYERUI_H
#define MAINPLAYERUI_H

#include <qvariant.h>
#include <qmainwindow.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QAction;
class QActionGroup;
class QToolBar;
class QPopupMenu;
class QFrame;
class QListView;
class QListViewItem;
class QWidgetStack;
class QWidget;
class QPushButton;
class QSlider;
class QLCDNumber;

class MainPlayerUI : public QMainWindow
{
    Q_OBJECT

public:
    MainPlayerUI( QWidget* parent = 0, const char* name = 0, WFlags fl = WType_TopLevel );
    ~MainPlayerUI();

    QFrame* listFrame;
    QListView* listView;
    QFrame* frame8;
    QFrame* frame7;
    QFrame* frame6;
    QFrame* pageFrame;
    QWidgetStack* widgetStack;
    QWidget* WStackPage;
    QFrame* buttonFrame;
    QPushButton* setupButton;
    QPushButton* runButton;
    QPushButton* pauseButton;
    QPushButton* continueButton;
    QPushButton* stopButton;
    QSlider* turnSlider;
    QLCDNumber* turnNumber;

public slots:
    virtual void startRun();
    virtual void pauseRun();
    virtual void continueRun();
    virtual void stopRun();
    virtual void initRun();
    virtual void showPage(QListViewItem*);

protected:
    QVBoxLayout* MainPlayerUILayout;
    QVBoxLayout* layout23;
    QHBoxLayout* layout22;
    QVBoxLayout* listFrameLayout;
    QVBoxLayout* layout19;
    QHBoxLayout* frame8Layout;
    QVBoxLayout* layout14;
    QHBoxLayout* frame7Layout;
    QHBoxLayout* frame6Layout;
    QVBoxLayout* pageFrameLayout;
    QHBoxLayout* buttonFrameLayout;
    QHBoxLayout* layout6;

protected slots:
    virtual void languageChange();

};

#endif // MAINPLAYERUI_H
