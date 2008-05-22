/****************************************************************************
** UAL::ROOT::MountainRangeViewer meta object code from reading C++ file 'MountainRangeViewer.hh'
**
** Created: Mon Oct 31 16:08:39 2005
**      by: The Qt MOC ($Id: moc_MountainRangeViewer.cc,v 1.4 2005/12/16 19:13:06 malitsky Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "MountainRangeViewer.hh"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *UAL::ROOT::MountainRangeViewer::className() const
{
    return "UAL::ROOT::MountainRangeViewer";
}

QMetaObject *UAL::ROOT::MountainRangeViewer::metaObj = 0;
static QMetaObjectCleanUp cleanUp_UAL__ROOT__MountainRangeViewer( "UAL::ROOT::MountainRangeViewer", &UAL::ROOT::MountainRangeViewer::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString UAL::ROOT::MountainRangeViewer::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::ROOT::MountainRangeViewer", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString UAL::ROOT::MountainRangeViewer::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::ROOT::MountainRangeViewer", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* UAL::ROOT::MountainRangeViewer::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = BasicViewer::staticMetaObject();
    metaObj = QMetaObject::new_metaobject(
	"UAL::ROOT::MountainRangeViewer", parentObject,
	0, 0,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_UAL__ROOT__MountainRangeViewer.setMetaObject( metaObj );
    return metaObj;
}

void* UAL::ROOT::MountainRangeViewer::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "UAL::ROOT::MountainRangeViewer" ) )
	return this;
    return BasicViewer::qt_cast( clname );
}

bool UAL::ROOT::MountainRangeViewer::qt_invoke( int _id, QUObject* _o )
{
    return BasicViewer::qt_invoke(_id,_o);
}

bool UAL::ROOT::MountainRangeViewer::qt_emit( int _id, QUObject* _o )
{
    return BasicViewer::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool UAL::ROOT::MountainRangeViewer::qt_property( int id, int f, QVariant* v)
{
    return BasicViewer::qt_property( id, f, v);
}

bool UAL::ROOT::MountainRangeViewer::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
