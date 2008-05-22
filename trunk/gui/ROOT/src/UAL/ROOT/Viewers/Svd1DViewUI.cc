/****************************************************************************
** Form implementation generated from reading ui file 'Svd1DViewUI.ui'
**
** Created: Thu Jun 2 06:20:42 2005
**      by: The User Interface Compiler ($Id: Svd1DViewUI.cc,v 1.2 2005/06/02 10:21:50 ual Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "Svd1DViewUI.hh"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qframe.h>
#include <qtabwidget.h>
#include <qtable.h>
#include <qlabel.h>
#include <qspinbox.h>
#include <qslider.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>

/*
 *  Constructs a Svd1DViewUI as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 */
Svd1DViewUI::Svd1DViewUI( QWidget* parent, const char* name, WFlags fl )
    : QWidget( parent, name, fl )
{
    if ( !name )
	setName( "Svd1DViewUI" );
    Svd1DViewUILayout = new QHBoxLayout( this, 11, 6, "Svd1DViewUILayout"); 

    layout23 = new QVBoxLayout( 0, 0, 3, "layout23"); 

    layout22 = new QHBoxLayout( 0, 0, 6, "layout22"); 

    pageFrame = new QFrame( this, "pageFrame" );
    pageFrame->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)7, 0, 0, pageFrame->sizePolicy().hasHeightForWidth() ) );
    pageFrame->setMinimumSize( QSize( 400, 0 ) );
    pageFrame->setFrameShape( QFrame::StyledPanel );
    pageFrame->setFrameShadow( QFrame::Plain );
    pageFrame->setLineWidth( 0 );
    pageFrameLayout = new QVBoxLayout( pageFrame, 0, 0, "pageFrameLayout"); 

    tabWidget = new QTabWidget( pageFrame, "tabWidget" );

    tab = new QWidget( tabWidget, "tab" );
    tabLayout = new QVBoxLayout( tab, 11, 6, "tabLayout"); 

    twissFrame = new QFrame( tab, "twissFrame" );
    twissFrame->setFrameShape( QFrame::StyledPanel );
    twissFrame->setFrameShadow( QFrame::Raised );
    twissFrame->setLineWidth( 0 );
    tabLayout->addWidget( twissFrame );
    tabWidget->insertTab( tab, QString("") );

    tab_2 = new QWidget( tabWidget, "tab_2" );
    tabLayout_2 = new QVBoxLayout( tab_2, 11, 6, "tabLayout_2"); 

    tableLayout = new QHBoxLayout( 0, 0, 6, "tableLayout"); 

    svdTable = new QTable( tab_2, "svdTable" );
    svdTable->setNumCols( svdTable->numCols() + 1 );
    svdTable->horizontalHeader()->setLabel( svdTable->numCols() - 1, tr( "singular value" ) );
    svdTable->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)3, (QSizePolicy::SizeType)7, 0, 0, svdTable->sizePolicy().hasHeightForWidth() ) );
    svdTable->setNumRows( 20 );
    svdTable->setNumCols( 1 );
    tableLayout->addWidget( svdTable );
    QSpacerItem* spacer = new QSpacerItem( 40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    tableLayout->addItem( spacer );
    tabLayout_2->addLayout( tableLayout );
    tabWidget->insertTab( tab_2, QString("") );

    TabPage = new QWidget( tabWidget, "TabPage" );
    TabPageLayout = new QVBoxLayout( TabPage, 11, 6, "TabPageLayout"); 

    layout6 = new QVBoxLayout( 0, 0, 6, "layout6"); 

    layout5 = new QHBoxLayout( 0, 0, 6, "layout5"); 

    setModeLabel = new QLabel( TabPage, "setModeLabel" );
    layout5->addWidget( setModeLabel );

    modeSpinBox = new QSpinBox( TabPage, "modeSpinBox" );
    layout5->addWidget( modeSpinBox );

    modeSlider = new QSlider( TabPage, "modeSlider" );
    modeSlider->setMinimumSize( QSize( 100, 0 ) );
    modeSlider->setOrientation( QSlider::Horizontal );
    layout5->addWidget( modeSlider );

    modeButton = new QPushButton( TabPage, "modeButton" );
    modeButton->setMinimumSize( QSize( 60, 0 ) );
    layout5->addWidget( modeButton );
    QSpacerItem* spacer_2 = new QSpacerItem( 40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout5->addItem( spacer_2 );
    layout6->addLayout( layout5 );

    svdFrame = new QFrame( TabPage, "svdFrame" );
    svdFrame->setFrameShape( QFrame::StyledPanel );
    svdFrame->setFrameShadow( QFrame::Raised );
    svdFrame->setLineWidth( 0 );
    layout6->addWidget( svdFrame );
    TabPageLayout->addLayout( layout6 );
    tabWidget->insertTab( TabPage, QString("") );
    pageFrameLayout->addWidget( tabWidget );
    layout22->addWidget( pageFrame );
    layout23->addLayout( layout22 );
    Svd1DViewUILayout->addLayout( layout23 );
    languageChange();
    resize( QSize(606, 432).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( modeSpinBox, SIGNAL( valueChanged(int) ), modeSlider, SLOT( setValue(int) ) );
    connect( modeSlider, SIGNAL( valueChanged(int) ), modeSpinBox, SLOT( setValue(int) ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
Svd1DViewUI::~Svd1DViewUI()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void Svd1DViewUI::languageChange()
{
    setCaption( tr( "Svd1DViewerUI" ) );
    tabWidget->changeTab( tab, tr( "Twiss" ) );
    svdTable->horizontalHeader()->setLabel( 0, tr( "singular value" ) );
    tabWidget->changeTab( tab_2, tr( "SVD eigenvalues" ) );
    setModeLabel->setText( tr( "Set mode" ) );
    modeButton->setText( tr( "OK" ) );
    tabWidget->changeTab( TabPage, tr( "SVD eigenvectors" ) );
}

