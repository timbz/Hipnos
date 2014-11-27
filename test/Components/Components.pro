QT       += testlib

TARGET = tst_components
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

DESTDIR = ../bin

DEFINES += SRCDIR=\\\"$$PWD/\\\"

unix {
    QMAKE_PRE_LINK = cp $$PWD/propagationdiff.sh $$DESTDIR
}

INCLUDEPATH += ../../Hipnos
HEADERS += \
    hipnoscomponenttest.h \
    fourieropticstest.h \
    gaussopticstest.h

SOURCES += \
    ../../Hipnos/common/pipelinecomponent.cpp \
    ../../Hipnos/common/pipelineconnection.cpp \
    ../../Hipnos/common/pipelinedata.cpp \
    ../../Hipnos/common/components/gaussianbeamsourcecomponent.cpp \
    ../../Hipnos/common/components/aperturecomponent.cpp \
    ../../Hipnos/common/components/propagationcomponent.cpp \
    ../../Hipnos/common/components/thinlenscomponent.cpp \
    ../../Hipnos/common/spectrum.cpp \
    main.cpp \
    mathmoc.cpp \
    hipnoscomponenttest.cpp \
    fourieropticstest.cpp \
    gaussopticstest.cpp

OTHER_FILES += \
    TestData/propagation-trans.csv \
    TestData/propagation.csv \
    TestData/propagation2.csv \
    TestData/gaussfill.csv \
    TestData/aperture-trans.csv \
    TestData/aperture.csv \
    TestData/lens-trans.csv \
    TestData/lens.csv \
    propagationdiff.sh

RESOURCES += \
    resources.qrc
