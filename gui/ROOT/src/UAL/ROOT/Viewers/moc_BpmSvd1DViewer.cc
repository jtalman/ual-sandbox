/****************************************************************************
** UAL::ROOT::BpmSvd1DViewer meta object code from reading C++ file 'BpmSvd1DViewer.hh'
**
** Created: Mon Oct 31 16:08:39 2005
**      by: The Qt MOC ($Id: moc_BpmSvd1DViewer.cc,v 1.4 2005/12/16 19:13:06 malitsky Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "BpmSvd1DViewer.hh"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *UAL::ROOT::BpmSvd1DViewer::className() const
{
    return "UAL::ROOT::BpmSvd1DViewer";
}

QMetaObject *UAL::ROOT::BpmSvd1DViewer::metaObj = 0;
static QMetaObjectCleanUp cleanUp_UAL__ROOT__BpmSvd1DViewer( "UAL::ROOT::BpmSvd1DViewer", &UAL::ROOT::BpmSvd1DViewer::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString UAL::ROOT::BpmSvd1DViewer::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::ROOT::BpmSvd1DViewer", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString UAL::ROOT::BpmSvd1DViewer::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::ROOT::BpmSvd1DViewer", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* UAL::ROOT::BpmSvd1DViewer::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = UAL::ROOT::BasicViewer::staticMetaObject();
    static const QUMethod slot_0 = {"startSVD", 0, 0 };
    static const QUMethod slot_1 = {"stopSVD", 0, 0 };
    static const QUParameter param_slot_2[] = {
	{ 0, &static_QUType_ptr, "QWidget", QUParameter::In }
    };
    static const QUMethod slot_2 = {"updateViewer", 1, param_slot_2 };
    static const QUMethod slot_3 = {"updateSvdGraphs", 0, 0 };
    static const QMetaData slot_tbl[] = {
	{ "startSVD()", &slot_0, QMetaData::Public },
	{ "stopSVD()", &slot_1, QMetaData::Public },
	{ "updateViewer(QWidget*)", &slot_2, QMetaData::Public },
	{ "updateSvdGraphs()", &slot_3, QMetaData::Public }
    };
    metaObj = QMetaObject::new_metaobject(
	"UAL::ROOT::BpmSvd1DViewer", parentObject,
	slot_tbl, 4,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_UAL__ROOT__BpmSvd1DViewer.setMetaObject( metaObj );
    return metaObj;
}

void* UAL::ROOT::BpmSvd1DViewer::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "UAL::ROOT::BpmSvd1DViewer" ) )
	return this;
    return BasicViewer::qt_cast( clname );
}

bool UAL::ROOT::BpmSvd1DViewer::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: startSVD(); break;
    case 1: stopSVD(); break;
    case 2: updateViewer((QWidget*)static_QUType_ptr.get(_o+1)); break;
    case 3: updateSvdGraphs(); break;
    default:
	return BasicViewer::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool UAL::ROOT::BpmSvd1DViewer::qt_emit( int _id, QUObject* _o )
{
    return BasicViewer::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool UAL::ROOT::BpmSvd1DViewer::qt_property( int id, int f, QVariant* v)
{
    return BasicViewer::qt_property( id, f, v);
}

bool UAL::ROOT::BpmSvd1DViewer::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
