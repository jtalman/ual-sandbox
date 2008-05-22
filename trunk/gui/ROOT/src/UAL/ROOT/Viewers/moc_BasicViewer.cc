/****************************************************************************
** UAL::ROOT::BasicViewer meta object code from reading C++ file 'BasicViewer.hh'
**
** Created: Mon Oct 31 16:08:39 2005
**      by: The Qt MOC ($Id: moc_BasicViewer.cc,v 1.5 2005/12/16 19:13:06 malitsky Exp $)
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

const char *UAL::ROOT::BasicViewer::className() const
{
    return "UAL::ROOT::BasicViewer";
}

QMetaObject *UAL::ROOT::BasicViewer::metaObj = 0;
static QMetaObjectCleanUp cleanUp_UAL__ROOT__BasicViewer( "UAL::ROOT::BasicViewer", &UAL::ROOT::BasicViewer::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString UAL::ROOT::BasicViewer::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::ROOT::BasicViewer", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString UAL::ROOT::BasicViewer::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::ROOT::BasicViewer", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* UAL::ROOT::BasicViewer::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = UAL::QT::BasicViewer::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ 0, &static_QUType_ptr, "TObject", QUParameter::In },
	{ 0, &static_QUType_ptr, "unsigned int", QUParameter::In },
	{ 0, &static_QUType_ptr, "TCanvas", QUParameter::In }
    };
    static const QUMethod slot_0 = {"processRootEvent", 3, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "processRootEvent(TObject*,unsigned int,TCanvas*)", &slot_0, QMetaData::Public }
    };
    metaObj = QMetaObject::new_metaobject(
	"UAL::ROOT::BasicViewer", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_UAL__ROOT__BasicViewer.setMetaObject( metaObj );
    return metaObj;
}

void* UAL::ROOT::BasicViewer::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "UAL::ROOT::BasicViewer" ) )
	return this;
    return UAL::QT::BasicViewer::qt_cast( clname );
}

bool UAL::ROOT::BasicViewer::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: processRootEvent((TObject*)static_QUType_ptr.get(_o+1),(unsigned int)(*((unsigned int*)static_QUType_ptr.get(_o+2))),(TCanvas*)static_QUType_ptr.get(_o+3)); break;
    default:
	return UAL::QT::BasicViewer::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool UAL::ROOT::BasicViewer::qt_emit( int _id, QUObject* _o )
{
    return UAL::QT::BasicViewer::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool UAL::ROOT::BasicViewer::qt_property( int id, int f, QVariant* v)
{
    return UAL::QT::BasicViewer::qt_property( id, f, v);
}

bool UAL::ROOT::BasicViewer::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
