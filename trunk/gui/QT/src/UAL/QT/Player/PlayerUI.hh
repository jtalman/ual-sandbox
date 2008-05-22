/****************************************************************************
** Form interface generated from reading ui file 'PlayerUI.ui'
**
** Created: Wed Apr 27 11:39:53 2005
**      by: The User Interface Compiler ($Id: PlayerUI.hh,v 1.1 2005/04/29 16:30:07 ual Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef PLAYERUI_H
#define PLAYERUI_H

#include <qvariant.h>
#include <qwidget.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QFrame;
class QListView;
class QListViewItem;
class QWidgetStack;
class QPushButton;
class QSlider;
class QLCDNumber;

class PlayerUI : public QWidget
{
    Q_OBJECT

public:
    PlayerUI( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
    ~PlayerUI();

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
    virtual void showPage(QListViewItem* item);
    virtual void stopRun();
    virtual void registerViewer(QWidget* viewer);
    virtual void continueRun();
    virtual void initRun();

protected:
    QHBoxLayout* PlayerUILayout;
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

#endif // PLAYERUI_H
