/****************************************************************************
** UAL::QT::BasicViewer meta object code from reading C++ file 'BasicViewer.hh'
**
** Created: Mon Oct 31 16:01:43 2005
**      by: The Qt MOC ($Id: moc_BasicViewer.cc,v 1.3 2005/12/16 19:12:15 malitsky Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "BasicViewer.hh"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *UAL::QT::BasicViewer::className() const
{
    return "UAL::QT::BasicViewer";
}

QMetaObject *UAL::QT::BasicViewer::metaObj = 0;
static QMetaObjectCleanUp cleanUp_UAL__QT__BasicViewer( "UAL::QT::BasicViewer", &UAL::QT::BasicViewer::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString UAL::QT::BasicViewer::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::QT::BasicViewer", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString UAL::QT::BasicViewer::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::QT::BasicViewer", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* UAL::QT::BasicViewer::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = QMainWindow::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ 0, &static_QUType_bool, 0, QUParameter::Out }
    };
    static const QUMethod slot_0 = {"saveAs", 1, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "saveAs()", &slot_0, QMetaData::Private }
    };
    metaObj = QMetaObject::new_metaobject(
	"UAL::QT::BasicViewer", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_UAL__QT__BasicViewer.setMetaObject( metaObj );
    return metaObj;
}

void* UAL::QT::BasicViewer::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "UAL::QT::BasicViewer" ) )
	return this;
    return QMainWindow::qt_cast( clname );
}

bool UAL::QT::BasicViewer::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: static_QUType_bool.set(_o,saveAs()); break;
    default:
	return QMainWindow::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool UAL::QT::BasicViewer::qt_emit( int _id, QUObject* _o )
{
    return QMainWindow::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool UAL::QT::BasicViewer::qt_property( int id, int f, QVariant* v)
{
    return QMainWindow::qt_property( id, f, v);
}

bool UAL::QT::BasicViewer::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
