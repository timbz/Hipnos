TEMPLATE        = lib
CONFIG         += plugin

HEADERS += \
    gslmath.h \
    gslvector.h \
    gslmatrix.h

SOURCES += \
    gslmath.cpp \
    gslmatrix.cpp \
    gslvector.cpp

INCLUDEPATH = ../../Hipnos

win32 {
    LIBS        += $$PWD/../../lib/gsl/1.8/libgslcblas.a
    LIBS        += $$PWD/../../lib/gsl/1.8/libgsl.a
    INCLUDEPATH += ../../include/gsl/1.8
}
unix {
    LIBS        += -lgslcblas -lgsl
}

TARGET          = $$qtLibraryTarget(HipnosGSLMathPlugin)
DESTDIR         = ../bin
