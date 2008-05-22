/****************************************************************************
** UAL::COIN3D::BunchViewer meta object code from reading C++ file 'BunchViewer.hh'
**
** Created: Thu Apr 28 15:33:56 2005
**      by: The Qt MOC ($Id: moc_BunchViewer.cc,v 1.1 2005/04/29 16:36:40 ual Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "BunchViewer.hh"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.2.0b2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *UAL::COIN3D::BunchViewer::className() const
{
    return "UAL::COIN3D::BunchViewer";
}

QMetaObject *UAL::COIN3D::BunchViewer::metaObj = 0;
static QMetaObjectCleanUp cleanUp_UAL__COIN3D__BunchViewer( "UAL::COIN3D::BunchViewer", &UAL::COIN3D::BunchViewer::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString UAL::COIN3D::BunchViewer::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::COIN3D::BunchViewer", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString UAL::COIN3D::BunchViewer::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::COIN3D::BunchViewer", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* UAL::COIN3D::BunchViewer::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = UAL::QT::BasicViewer::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"UAL::COIN3D::BunchViewer", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_UAL__COIN3D__BunchViewer.setMetaObject( metaObj );
    return metaObj;
}

void* UAL::COIN3D::BunchViewer::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "UAL::COIN3D::BunchViewer" ) )
	return this;
    return BasicViewer::qt_cast( clname );
}

bool UAL::COIN3D::BunchViewer::qt_invoke( int _id, QUObject* _o )
{
    return BasicViewer::qt_invoke(_id,_o);
}

bool UAL::COIN3D::BunchViewer::qt_emit( int _id, QUObject* _o )
{
    return BasicViewer::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool UAL::COIN3D::BunchViewer::qt_property( int id, int f, QVariant* v)
{
    return BasicViewer::qt_property( id, f, v);
}

bool UAL::COIN3D::BunchViewer::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
