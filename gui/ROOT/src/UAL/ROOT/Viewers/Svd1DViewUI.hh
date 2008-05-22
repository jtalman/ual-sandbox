/****************************************************************************
** Form interface generated from reading ui file 'Svd1DViewUI.ui'
**
** Created: Thu Jun 2 06:20:42 2005
**      by: The User Interface Compiler ($Id: Svd1DViewUI.hh,v 1.2 2005/06/02 10:21:50 ual Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#ifndef SVD1DVIEWUI_H
#define SVD1DVIEWUI_H

#include <qvariant.h>
#include <qwidget.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QFrame;
class QTabWidget;
class QTable;
class QLabel;
class QSpinBox;
class QSlider;
class QPushButton;

class Svd1DViewUI : public QWidget
{
    Q_OBJECT

public:
    Svd1DViewUI( QWidget* parent = 0, const char* name = 0, WFlags fl = 0 );
    ~Svd1DViewUI();

    QFrame* pageFrame;
    QTabWidget* tabWidget;
    QWidget* tab;
    QFrame* twissFrame;
    QWidget* tab_2;
    QTable* svdTable;
    QWidget* TabPage;
    QLabel* setModeLabel;
    QSpinBox* modeSpinBox;
    QSlider* modeSlider;
    QPushButton* modeButton;
    QFrame* svdFrame;

protected:
    QHBoxLayout* Svd1DViewUILayout;
    QVBoxLayout* layout23;
    QHBoxLayout* layout22;
    QVBoxLayout* pageFrameLayout;
    QVBoxLayout* tabLayout;
    QVBoxLayout* tabLayout_2;
    QHBoxLayout* tableLayout;
    QVBoxLayout* TabPageLayout;
    QVBoxLayout* layout6;
    QHBoxLayout* layout5;

protected slots:
    virtual void languageChange();

};

#endif // SVD1DVIEWUI_H
