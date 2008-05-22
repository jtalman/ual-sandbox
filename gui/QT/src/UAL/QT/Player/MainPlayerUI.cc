/****************************************************************************
** Form implementation generated from reading ui file 'MainPlayerUI.ui'
**
** Created: Thu May 26 21:00:54 2005
**      by: The User Interface Compiler ($Id: MainPlayerUI.cc,v 1.1 2005/05/31 17:11:42 ual Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "MainPlayerUI.hh"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qframe.h>
#include <qheader.h>
#include <qlistview.h>
#include <qwidgetstack.h>
#include <qwidget.h>
#include <qslider.h>
#include <qlcdnumber.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qaction.h>
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qtoolbar.h>

/*
 *  Constructs a MainPlayerUI as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
MainPlayerUI::MainPlayerUI( QWidget* parent, const char* name, WFlags fl )
    : QMainWindow( parent, name, fl )
{
    (void)statusBar();
    if ( !name )
	setName( "MainPlayerUI" );
    setCentralWidget( new QWidget( this, "qt_central_widget" ) );
    MainPlayerUILayout = new QVBoxLayout( centralWidget(), 11, 6, "MainPlayerUILayout"); 

    layout23 = new QVBoxLayout( 0, 0, 3, "layout23"); 

    layout22 = new QHBoxLayout( 0, 0, 6, "layout22"); 

    listFrame = new QFrame( centralWidget(), "listFrame" );
    listFrame->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)5, 0, 0, listFrame->sizePolicy().hasHeightForWidth() ) );
    listFrame->setMinimumSize( QSize( 160, 0 ) );
    listFrame->setMaximumSize( QSize( 150, 32767 ) );
    listFrame->setFrameShape( QFrame::StyledPanel );
    listFrame->setFrameShadow( QFrame::Plain );
    listFrame->setLineWidth( 1 );
    listFrameLayout = new QVBoxLayout( listFrame, 5, 0, "listFrameLayout"); 

    layout19 = new QVBoxLayout( 0, 0, 6, "layout19"); 

    listView = new QListView( listFrame, "listView" );
    listView->addColumn( tr( "Tools" ) );
    listView->header()->setResizeEnabled( FALSE, listView->header()->count() - 1 );
    listView->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)7, 0, 0, listView->sizePolicy().hasHeightForWidth() ) );
    listView->setMinimumSize( QSize( 150, 0 ) );
    listView->setMaximumSize( QSize( 150, 32767 ) );
    listView->setFrameShape( QListView::Panel );
    listView->setFrameShadow( QListView::Raised );
    listView->setLineWidth( 0 );
    listView->setMidLineWidth( 1 );
    listView->setResizePolicy( QScrollView::Manual );
    listView->setHScrollBarMode( QListView::Auto );
    listView->setRootIsDecorated( TRUE );
    layout19->addWidget( listView );

    frame8 = new QFrame( listFrame, "frame8" );
    frame8->setAcceptDrops( FALSE );
    frame8->setFrameShape( QFrame::StyledPanel );
    frame8->setFrameShadow( QFrame::Raised );
    frame8Layout = new QHBoxLayout( frame8, 4, 0, "frame8Layout"); 

    layout14 = new QVBoxLayout( 0, 0, 0, "layout14"); 

    frame7 = new QFrame( frame8, "frame7" );
    frame7->setFrameShape( QFrame::StyledPanel );
    frame7->setFrameShadow( QFrame::Plain );
    frame7->setLineWidth( 0 );
    frame7Layout = new QHBoxLayout( frame7, 0, 0, "frame7Layout"); 
    layout14->addWidget( frame7 );

    frame6 = new QFrame( frame8, "frame6" );
    frame6->setAcceptDrops( FALSE );
    frame6->setFrameShape( QFrame::StyledPanel );
    frame6->setFrameShadow( QFrame::Plain );
    frame6->setLineWidth( 0 );
    frame6Layout = new QHBoxLayout( frame6, 9, 0, "frame6Layout"); 
    layout14->addWidget( frame6 );
    frame8Layout->addLayout( layout14 );
    layout19->addWidget( frame8 );
    listFrameLayout->addLayout( layout19 );
    layout22->addWidget( listFrame );

    pageFrame = new QFrame( centralWidget(), "pageFrame" );
    pageFrame->setMinimumSize( QSize( 400, 0 ) );
    pageFrame->setFrameShape( QFrame::StyledPanel );
    pageFrame->setFrameShadow( QFrame::Plain );
    pageFrameLayout = new QVBoxLayout( pageFrame, 11, 1, "pageFrameLayout"); 

    widgetStack = new QWidgetStack( pageFrame, "widgetStack" );

    WStackPage = new QWidget( widgetStack, "WStackPage" );
    widgetStack->addWidget( WStackPage, 0 );
    pageFrameLayout->addWidget( widgetStack );
    layout22->addWidget( pageFrame );
    layout23->addLayout( layout22 );

    buttonFrame = new QFrame( centralWidget(), "buttonFrame" );
    buttonFrame->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, buttonFrame->sizePolicy().hasHeightForWidth() ) );
    buttonFrame->setMinimumSize( QSize( 0, 0 ) );
    buttonFrame->setFrameShape( QFrame::StyledPanel );
    buttonFrame->setFrameShadow( QFrame::Plain );
    buttonFrameLayout = new QHBoxLayout( buttonFrame, 2, 6, "buttonFrameLayout"); 

    layout6 = new QHBoxLayout( 0, 0, 6, "layout6"); 

    setupButton = new QPushButton( buttonFrame, "setupButton" );
    setupButton->setMinimumSize( QSize( 50, 25 ) );
    setupButton->setMaximumSize( QSize( 50, 25 ) );
    layout6->addWidget( setupButton );

    runButton = new QPushButton( buttonFrame, "runButton" );
    runButton->setMaximumSize( QSize( 40, 25 ) );
    layout6->addWidget( runButton );

    pauseButton = new QPushButton( buttonFrame, "pauseButton" );
    pauseButton->setMaximumSize( QSize( 60, 25 ) );
    layout6->addWidget( pauseButton );

    continueButton = new QPushButton( buttonFrame, "continueButton" );
    continueButton->setMaximumSize( QSize( 60, 25 ) );
    layout6->addWidget( continueButton );

    stopButton = new QPushButton( buttonFrame, "stopButton" );
    stopButton->setMinimumSize( QSize( 0, 0 ) );
    stopButton->setMaximumSize( QSize( 40, 25 ) );
    layout6->addWidget( stopButton );

    turnSlider = new QSlider( buttonFrame, "turnSlider" );
    turnSlider->setMaximumSize( QSize( 32767, 25 ) );
    turnSlider->setOrientation( QSlider::Horizontal );
    layout6->addWidget( turnSlider );

    turnNumber = new QLCDNumber( buttonFrame, "turnNumber" );
    turnNumber->setMaximumSize( QSize( 32767, 25 ) );
    layout6->addWidget( turnNumber );
    QSpacerItem* spacer = new QSpacerItem( 100, 25, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout6->addItem( spacer );
    buttonFrameLayout->addLayout( layout6 );
    layout23->addWidget( buttonFrame );
    MainPlayerUILayout->addLayout( layout23 );

    // toolbars

    languageChange();
    resize( QSize(600, 482).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( runButton, SIGNAL( clicked() ), this, SLOT( startRun() ) );
    connect( pauseButton, SIGNAL( clicked() ), this, SLOT( pauseRun() ) );
    connect( continueButton, SIGNAL( clicked() ), this, SLOT( continueRun() ) );
    connect( stopButton, SIGNAL( clicked() ), this, SLOT( stopRun() ) );
    connect( setupButton, SIGNAL( clicked() ), this, SLOT( initRun() ) );
    connect( listView, SIGNAL( doubleClicked(QListViewItem*) ), this, SLOT( showPage(QListViewItem*) ) );
    connect( turnSlider, SIGNAL( valueChanged(int) ), turnNumber, SLOT( display(int) ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
MainPlayerUI::~MainPlayerUI()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void MainPlayerUI::languageChange()
{
    setCaption( tr( "MainPlayerUI" ) );
    listView->header()->setLabel( 0, tr( "Tools" ) );
    setupButton->setText( tr( "Setup" ) );
    runButton->setText( tr( "Run" ) );
    pauseButton->setText( tr( "Pause" ) );
    continueButton->setText( tr( "Continue" ) );
    stopButton->setText( tr( "Stop" ) );
}

void MainPlayerUI::startRun()
{
    qWarning( "MainPlayerUI::startRun(): Not implemented yet" );
}

void MainPlayerUI::pauseRun()
{
    qWarning( "MainPlayerUI::pauseRun(): Not implemented yet" );
}

void MainPlayerUI::continueRun()
{
    qWarning( "MainPlayerUI::continueRun(): Not implemented yet" );
}

void MainPlayerUI::stopRun()
{
    qWarning( "MainPlayerUI::stopRun(): Not implemented yet" );
}

void MainPlayerUI::initRun()
{
    qWarning( "MainPlayerUI::initRun(): Not implemented yet" );
}

void MainPlayerUI::showPage(QListViewItem*)
{
    qWarning( "MainPlayerUI::showPage(QListViewItem*): Not implemented yet" );
}

