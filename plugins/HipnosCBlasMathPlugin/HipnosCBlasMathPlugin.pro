TEMPLATE        = lib
CONFIG         += plugin

HEADERS += \
    cblasvector.h \
    cblasmatrix.h \
    cblasmath.h

SOURCES += \
    cblasvector.cpp \
    cblasmatrix.cpp \
    cblasmath.cpp

INCLUDEPATH = ../../Hipnos

win32 {
}
unix {
    LIBS        += -lblas -lfftw3 -lm
}

TARGET          = $$qtLibraryTarget(HipnosCBlasMathPlugin)
DESTDIR         = ../bin
