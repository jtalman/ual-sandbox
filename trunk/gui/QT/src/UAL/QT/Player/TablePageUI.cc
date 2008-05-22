/****************************************************************************
** Form implementation generated from reading ui file 'TablePageUI.ui'
**
** Created: Wed Apr 27 11:40:15 2005
**      by: The User Interface Compiler ($Id: TablePageUI.cc,v 1.1 2005/04/29 16:30:08 ual Exp $)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/

#include "TablePageUI.hh"

#include <qvariant.h>
#include <qpushbutton.h>
#include <qframe.h>
#include <qlabel.h>
#include <qtable.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>

/*
 *  Constructs a TablePageUI as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 */
TablePageUI::TablePageUI( QWidget* parent, const char* name, WFlags fl )
    : QWidget( parent, name, fl )
{
    if ( !name )
	setName( "TablePageUI" );
    TablePageUILayout = new QHBoxLayout( this, 0, 6, "TablePageUILayout"); 

    frame3 = new QFrame( this, "frame3" );
    frame3->setFrameShape( QFrame::Panel );
    frame3->setFrameShadow( QFrame::Plain );
    frame3->setLineWidth( 0 );
    frame3Layout = new QVBoxLayout( frame3, 0, 6, "frame3Layout"); 

    frame23 = new QFrame( frame3, "frame23" );
    frame23->setFrameShape( QFrame::StyledPanel );
    frame23->setFrameShadow( QFrame::Raised );
    frame23Layout = new QVBoxLayout( frame23, 5, 5, "frame23Layout"); 

    label = new QLabel( frame23, "label" );
    label->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, label->sizePolicy().hasHeightForWidth() ) );
    label->setFrameShape( QLabel::Box );
    label->setFrameShadow( QLabel::Raised );
    label->setMargin( 2 );
    frame23Layout->addWidget( label );

    table = new QTable( frame23, "table" );
    table->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)7, (QSizePolicy::SizeType)5, 0, 0, table->sizePolicy().hasHeightForWidth() ) );
    table->setResizePolicy( QTable::Default );
    table->setVScrollBarMode( QTable::Auto );
    table->setHScrollBarMode( QTable::Auto );
    table->setNumRows( 0 );
    table->setNumCols( 0 );
    table->setColumnMovingEnabled( TRUE );
    table->setSorting( FALSE );
    table->setFocusStyle( QTable::SpreadSheet );
    frame23Layout->addWidget( table );
    frame3Layout->addWidget( frame23 );

    layout3 = new QHBoxLayout( 0, 0, 6, "layout3"); 
    QSpacerItem* spacer = new QSpacerItem( 40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
    layout3->addItem( spacer );
    frame3Layout->addLayout( layout3 );
    QSpacerItem* spacer_2 = new QSpacerItem( 20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding );
    frame3Layout->addItem( spacer_2 );
    TablePageUILayout->addWidget( frame3 );
    languageChange();
    resize( QSize(606, 569).expandedTo(minimumSizeHint()) );
    clearWState( WState_Polished );

    // signals and slots connections
    connect( table, SIGNAL( valueChanged(int,int) ), this, SLOT( setValue(int,int) ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
TablePageUI::~TablePageUI()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void TablePageUI::languageChange()
{
    setCaption( tr( "TablePageUI" ) );
    label->setText( tr( "Parameters" ) );
}

void TablePageUI::setValue(int,int)
{
    qWarning( "TablePageUI::setValue(int,int): Not implemented yet" );
}

void TablePageUI::activateChanges()
{
    qWarning( "TablePageUI::activateChanges(): Not implemented yet" );
}

