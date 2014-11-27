#-------------------------------------------------
#
# Project created by QtCreator 2012-09-11T14:38:57
#
#-------------------------------------------------

QT       += testlib

TARGET = pipelinebenchmark
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

DESTDIR = ../bin

INCLUDEPATH += ../../Hipnos

unix {
    QMAKE_PRE_LINK = cp $$PWD/gencharts.sh $$DESTDIR
}

SOURCES += pipelinebenchmark.cpp \
    ../../Hipnos/common/pipelinecomponent.cpp \
    ../../Hipnos/common/pipelineconnection.cpp \
    ../../Hipnos/common/pipelinedata.cpp \
    ../../Hipnos/common/components/gaussianbeamsourcecomponent.cpp \
    ../../Hipnos/common/components/aperturecomponent.cpp \
    ../../Hipnos/common/components/propagationcomponent.cpp \
    ../../Hipnos/common/components/thinlenscomponent.cpp \
    ../../Hipnos/common/spectrum.cpp \
    ../../Hipnos/common/pipeline.cpp \
    mathmoc.cpp

DEFINES += SRCDIR=\\\"$$PWD/\\\"

HEADERS += \
    pipelinebenchmark.h

OTHER_FILES += \
    gencharts.sh
