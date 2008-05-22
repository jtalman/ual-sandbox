/****************************************************************************
** PlayerUI meta object code from reading C++ file 'PlayerUI.hh'
**
** Created: Mon Oct 31 16:01:43 2005
**      by: The Qt MOC ($Id: moc_PlayerUI.cc,v 1.2 2005/12/16 19:12:15 malitsky Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "PlayerUI.hh"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *PlayerUI::className() const
{
    return "PlayerUI";
}

QMetaObject *PlayerUI::metaObj = 0;
static QMetaObjectCleanUp cleanUp_PlayerUI( "PlayerUI", &PlayerUI::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString PlayerUI::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "PlayerUI", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString PlayerUI::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "PlayerUI", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* PlayerUI::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = QWidget::staticMetaObject();
    static const QUMethod slot_0 = {"startRun", 0, 0 };
    static const QUMethod slot_1 = {"pauseRun", 0, 0 };
    static const QUParameter param_slot_2[] = {
	{ "item", &static_QUType_ptr, "QListViewItem", QUParameter::In }
    };
    static const QUMethod slot_2 = {"showPage", 1, param_slot_2 };
    static const QUMethod slot_3 = {"stopRun", 0, 0 };
    static const QUParameter param_slot_4[] = {
	{ "viewer", &static_QUType_ptr, "QWidget", QUParameter::In }
    };
    static const QUMethod slot_4 = {"registerViewer", 1, param_slot_4 };
    static const QUMethod slot_5 = {"continueRun", 0, 0 };
    static const QUMethod slot_6 = {"initRun", 0, 0 };
    static const QUMethod slot_7 = {"languageChange", 0, 0 };
    static const QMetaData slot_tbl[] = {
	{ "startRun()", &slot_0, QMetaData::Public },
	{ "pauseRun()", &slot_1, QMetaData::Public },
	{ "showPage(QListViewItem*)", &slot_2, QMetaData::Public },
	{ "stopRun()", &slot_3, QMetaData::Public },
	{ "registerViewer(QWidget*)", &slot_4, QMetaData::Public },
	{ "continueRun()", &slot_5, QMetaData::Public },
	{ "initRun()", &slot_6, QMetaData::Public },
	{ "languageChange()", &slot_7, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"PlayerUI", parentObject,
	slot_tbl, 8,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_PlayerUI.setMetaObject( metaObj );
    return metaObj;
}

void* PlayerUI::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "PlayerUI" ) )
	return this;
    return QWidget::qt_cast( clname );
}

bool PlayerUI::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: startRun(); break;
    case 1: pauseRun(); break;
    case 2: showPage((QListViewItem*)static_QUType_ptr.get(_o+1)); break;
    case 3: stopRun(); break;
    case 4: registerViewer((QWidget*)static_QUType_ptr.get(_o+1)); break;
    case 5: continueRun(); break;
    case 6: initRun(); break;
    case 7: languageChange(); break;
    default:
	return QWidget::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool PlayerUI::qt_emit( int _id, QUObject* _o )
{
    return QWidget::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool PlayerUI::qt_property( int id, int f, QVariant* v)
{
    return QWidget::qt_property( id, f, v);
}

bool PlayerUI::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
