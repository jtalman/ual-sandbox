/****************************************************************************
** UAL::USPAS::RFEditor meta object code from reading C++ file 'RFEditor.hh'
**
** Created: Mon Oct 31 16:08:39 2005
**      by: The Qt MOC ($Id: moc_RFEditor.cc,v 1.3 2005/12/16 19:13:06 malitsky Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "RFEditor.hh"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *UAL::USPAS::RFEditor::className() const
{
    return "UAL::USPAS::RFEditor";
}

QMetaObject *UAL::USPAS::RFEditor::metaObj = 0;
static QMetaObjectCleanUp cleanUp_UAL__USPAS__RFEditor( "UAL::USPAS::RFEditor", &UAL::USPAS::RFEditor::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString UAL::USPAS::RFEditor::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::USPAS::RFEditor", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString UAL::USPAS::RFEditor::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "UAL::USPAS::RFEditor", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* UAL::USPAS::RFEditor::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = UAL::QT::BasicEditor::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ "row", &static_QUType_int, 0, QUParameter::In },
	{ "col", &static_QUType_int, 0, QUParameter::In }
    };
    static const QUMethod slot_0 = {"setValue", 2, param_slot_0 };
    static const QMetaData slot_tbl[] = {
	{ "setValue(int,int)", &slot_0, QMetaData::Public }
    };
    metaObj = QMetaObject::new_metaobject(
	"UAL::USPAS::RFEditor", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_UAL__USPAS__RFEditor.setMetaObject( metaObj );
    return metaObj;
}

void* UAL::USPAS::RFEditor::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "UAL::USPAS::RFEditor" ) )
	return this;
    return BasicEditor::qt_cast( clname );
}

bool UAL::USPAS::RFEditor::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: setValue((int)static_QUType_int.get(_o+1),(int)static_QUType_int.get(_o+2)); break;
    default:
	return BasicEditor::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool UAL::USPAS::RFEditor::qt_emit( int _id, QUObject* _o )
{
    return BasicEditor::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool UAL::USPAS::RFEditor::qt_property( int id, int f, QVariant* v)
{
    return BasicEditor::qt_property( id, f, v);
}

bool UAL::USPAS::RFEditor::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
