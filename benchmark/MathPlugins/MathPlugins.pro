QT       += testlib

QT       -= gui

TARGET = mathpluginsbenchmark
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

DESTDIR = ../bin

unix {
    QMAKE_PRE_LINK = cd $$PWD && chmod +x copy_scripts.sh && ./copy_scripts.sh $$OUT_PWD/$$DESTDIR
}

INCLUDEPATH = ../../Hipnos

SOURCES += \
    mathpluginsbenchmark.cpp

OTHER_FILES += \
    scripts/3dplot/* \
    scripts/2dplot/* \
    copy_scripts.sh

HEADERS += \
    mathpluginsbenchmark.h

#   TODO:
#   AMD APPML: OpenCL BLAS + FFT (http://developer.amd.com/libraries/appmathlibs/pages/default.aspx)
#   MAGMA: BLAS on CPU and GPU depending on size
#   INTEL MKL: BLAS + FFT CPU Multithreaded (http://software.intel.com/en-us/articles/intel-mkl/)
