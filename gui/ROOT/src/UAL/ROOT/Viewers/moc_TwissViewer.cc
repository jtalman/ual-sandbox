/****************************************************************************
** UAL::ROOT::TwissViewer meta object code from reading C++ file 'TwissViewer.hh'
**
** Created: Mon Oct 31 16:08:39 2005
**      by: The Qt MOC ($Id: moc_TwissViewer.cc,v 1.4 2005/12/16 19:13:06 malitsky Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "TwissViewer.hh"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *UAL::ROOT::TwissViewer::className() const
{
    return "UAL::ROOT::TwissViewer";
}

QMetaObject *UAL::ROOT::TwissViewer::metaObj = 0;
static QMetaObjectCleanUp cleanUp_UAL__ROOT__TwissViewer( "UAL::ROOT::TwissViewer", &UAL::ROOT::TwissViewer::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString UAL::ROOT::TwissViewer::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::ROOT::TwissViewer", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString UAL::ROOT::TwissViewer::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::ROOT::TwissViewer", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* UAL::ROOT::TwissViewer::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = BasicViewer::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ 0, &static_QUType_bool, 0, QUParameter::Out }
    };
    static const QUMethod slot_0 = {"writeTo", 1, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "writeTo()", &slot_0, QMetaData::Private }
    };
    metaObj = QMetaObject::new_metaobject(
	"UAL::ROOT::TwissViewer", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_UAL__ROOT__TwissViewer.setMetaObject( metaObj );
    return metaObj;
}

void* UAL::ROOT::TwissViewer::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "UAL::ROOT::TwissViewer" ) )
	return this;
    return BasicViewer::qt_cast( clname );
}

bool UAL::ROOT::TwissViewer::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: static_QUType_bool.set(_o,writeTo()); break;
    default:
	return BasicViewer::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool UAL::ROOT::TwissViewer::qt_emit( int _id, QUObject* _o )
{
    return BasicViewer::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool UAL::ROOT::TwissViewer::qt_property( int id, int f, QVariant* v)
{
    return BasicViewer::qt_property( id, f, v);
}

bool UAL::ROOT::TwissViewer::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
