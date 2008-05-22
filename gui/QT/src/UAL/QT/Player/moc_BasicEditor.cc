/****************************************************************************
** UAL::QT::BasicEditor meta object code from reading C++ file 'BasicEditor.hh'
**
** Created: Fri Dec 23 07:06:07 2005
**      by: The Qt MOC ($Id: moc_BasicEditor.cc,v 1.4 2006/01/12 22:11:22 malitsky Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "BasicEditor.hh"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *UAL::QT::BasicEditor::className() const
{
    return "UAL::QT::BasicEditor";
}

QMetaObject *UAL::QT::BasicEditor::metaObj = 0;
static QMetaObjectCleanUp cleanUp_UAL__QT__BasicEditor( "UAL::QT::BasicEditor", &UAL::QT::BasicEditor::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString UAL::QT::BasicEditor::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::QT::BasicEditor", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString UAL::QT::BasicEditor::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::QT::BasicEditor", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* UAL::QT::BasicEditor::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = TablePageUI::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"UAL::QT::BasicEditor", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_UAL__QT__BasicEditor.setMetaObject( metaObj );
    return metaObj;
}

void* UAL::QT::BasicEditor::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "UAL::QT::BasicEditor" ) )
	return this;
    return TablePageUI::qt_cast( clname );
}

bool UAL::QT::BasicEditor::qt_invoke( int _id, QUObject* _o )
{
    return TablePageUI::qt_invoke(_id,_o);
}

bool UAL::QT::BasicEditor::qt_emit( int _id, QUObject* _o )
{
    return TablePageUI::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool UAL::QT::BasicEditor::qt_property( int id, int f, QVariant* v)
{
    return TablePageUI::qt_property( id, f, v);
}

bool UAL::QT::BasicEditor::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
