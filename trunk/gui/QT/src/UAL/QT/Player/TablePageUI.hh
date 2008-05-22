/****************************************************************************
** Form interface generated from reading ui file 'TablePageUI.ui'
**
** Created: Wed Apr 27 11:40:14 2005
**      by: The User Interface Compiler ($Id: TablePageUI.hh,v 1.1 2005/04/29 16:30:08 ual Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef TABLEPAGEUI_H
#define TABLEPAGEUI_H

#include <qvariant.h>
#include <qwidget.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QFrame;
class QLabel;
class QTable;

class TablePageUI : public QWidget
{
    Q_OBJECT

public:
    TablePageUI( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
    ~TablePageUI();

    QFrame* frame3;
    QFrame* frame23;
    QLabel* label;
    QTable* table;

public slots:
    virtual void setValue( int row, int col );
    virtual void activateChanges();

protected:
    QHBoxLayout* TablePageUILayout;
    QVBoxLayout* frame3Layout;
    QVBoxLayout* frame23Layout;
    QHBoxLayout* layout3;

protected slots:
    virtual void languageChange();

};

#endif // TABLEPAGEUI_H
