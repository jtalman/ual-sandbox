/****************************************************************************
** Svd1DViewUI meta object code from reading C++ file 'Svd1DViewUI.hh'
**
** Created: Mon Oct 31 16:08:39 2005
**      by: The Qt MOC ($Id: moc_Svd1DViewUI.cc,v 1.4 2005/12/16 19:13:06 malitsky Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#undef QT_NO_COMPAT
#include "Svd1DViewUI.hh"
#include <qmetaobject.h>
#include <qapplication.h>

#include <private/qucomextra_p.h>
#if !defined(Q_MOC_OUTPUT_REVISION) || (Q_MOC_OUTPUT_REVISION != 26)
#error "This file was generated using the moc from 3.3.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

const char *Svd1DViewUI::className() const
{
    return "Svd1DViewUI";
}

QMetaObject *Svd1DViewUI::metaObj = 0;
static QMetaObjectCleanUp cleanUp_Svd1DViewUI( "Svd1DViewUI", &Svd1DViewUI::staticMetaObject );

#ifndef QT_NO_TRANSLATION
QString Svd1DViewUI::tr( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "Svd1DViewUI", s, c, QApplication::DefaultCodec );
    else
	return QString::fromLatin1( s );
}
#ifndef QT_NO_TRANSLATION_UTF8
QString Svd1DViewUI::trUtf8( const char *s, const char *c )
{
    if ( qApp )
	return qApp->translate( "Svd1DViewUI", s, c, QApplication::UnicodeUTF8 );
    else
	return QString::fromUtf8( s );
}
#endif // QT_NO_TRANSLATION_UTF8

#endif // QT_NO_TRANSLATION

QMetaObject* Svd1DViewUI::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    QMetaObject* parentObject = QWidget::staticMetaObject();
    static const QUMethod slot_0 = {"languageChange", 0, 0 };
    static const QMetaData slot_tbl[] = {
	{ "languageChange()", &slot_0, QMetaData::Protected }
    };
    metaObj = QMetaObject::new_metaobject(
	"Svd1DViewUI", parentObject,
	slot_tbl, 1,
	0, 0,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    cleanUp_Svd1DViewUI.setMetaObject( metaObj );
    return metaObj;
}

void* Svd1DViewUI::qt_cast( const char* clname )
{
    if ( !qstrcmp( clname, "Svd1DViewUI" ) )
	return this;
    return QWidget::qt_cast( clname );
}

bool Svd1DViewUI::qt_invoke( int _id, QUObject* _o )
{
    switch ( _id - staticMetaObject()->slotOffset() ) {
    case 0: languageChange(); break;
    default:
	return QWidget::qt_invoke( _id, _o );
    }
    return TRUE;
}

bool Svd1DViewUI::qt_emit( int _id, QUObject* _o )
{
    return QWidget::qt_emit(_id,_o);
}
#ifndef QT_NO_PROPERTIES

bool Svd1DViewUI::qt_property( int id, int f, QVariant* v)
{
    return QWidget::qt_property( id, f, v);
}

bool Svd1DViewUI::qt_static_property( QObject* , int , int , QVariant* ){ return FALSE; }
#endif // QT_NO_PROPERTIES
