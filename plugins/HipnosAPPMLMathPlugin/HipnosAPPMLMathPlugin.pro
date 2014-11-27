TEMPLATE        = lib
CONFIG         += plugin

HEADERS += \
    appmlvector.h \
    appmlmatrix.h \
    appmlmath.h \
    openclplatformanddevicedialog.h \
    opencldevice.h

SOURCES += \
    appmlvector.cpp \
    appmlmatrix.cpp \
    appmlmath.cpp \
    openclplatformanddevicedialog.cpp

INCLUDEPATH = ../../Hipnos

win32 {

}
unix {
    LIBS        += -L/opt/clAmdBlas-1.8.269/lib64 -lclAmdBlas
    LIBS        += -L/opt/clAmdFft-1.8.276/lib64 -lclAmdFft.Runtime
    INCLUDEPATH += /opt/clAmdBlas-1.8.269/include
    INCLUDEPATH += /opt/clAmdFft-1.8.276/include
    INCLUDEPATH += /usr/local/cuda/include
}

TARGET          = $$qtLibraryTarget(HipnosAPPMLMathPlugin)
DESTDIR         = ../bin

OTHER_FILES += \
    kernels.cl

RESOURCES += \
    resources.qrc
