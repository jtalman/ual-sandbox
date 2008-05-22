/****************************************************************************
** TablePageUI meta object code from reading C++ file 'TablePageUI.hh'
**
** Created: Mon Oct 31 16:01:43 2005
**      by: The Qt MOC ($Id: moc_TablePageUI.cc,v 1.2 2005/12/16 19:12:15 malitsky Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "TablePageUI.hh"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *TablePageUI::className() const
{
    return "TablePageUI";
}

QMetaObject *TablePageUI::metaObj = 0;
static QMetaObjectCleanUp cleanUp_TablePageUI( "TablePageUI", &TablePageUI::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString TablePageUI::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "TablePageUI", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString TablePageUI::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "TablePageUI", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* TablePageUI::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = QWidget::staticMetaObject();
    static const QUParameter param_slot_0[] = {
	{ "row", &static_QUType_int, 0, QUParameter::In },
	{ "col", &static_QUType_int, 0, QUParameter::In }
    };
    static const QUMethod slot_0 = {"setValue", 2, param_slot_0 };
    static const QUMethod slot_1 = {"activateChanges", 0, 0 };
    static const QUMethod slot_2 = {"languageChange", 0, 0 };
    static const QMetaData slot_tbl[] = {
	{ "setValue(int,int)", &slot_0, QMetaData::Public },
	{ "activateChanges()", &slot_1, QMetaData::Public },
	{ "languageChange()", &slot_2, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"TablePageUI", parentObject,
	slot_tbl, 3,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_TablePageUI.setMetaObject( metaObj );
    return metaObj;
}

void* TablePageUI::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "TablePageUI" ) )
	return this;
    return QWidget::qt_cast( clname );
}

bool TablePageUI::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: setValue((int)static_QUType_int.get(_o+1),(int)static_QUType_int.get(_o+2)); break;
    case 1: activateChanges(); break;
    case 2: languageChange(); break;
    default:
	return QWidget::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool TablePageUI::qt_emit( int _id, QUObject* _o )
{
    return QWidget::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool TablePageUI::qt_property( int id, int f, QVariant* v)
{
    return QWidget::qt_property( id, f, v);
}

bool TablePageUI::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
