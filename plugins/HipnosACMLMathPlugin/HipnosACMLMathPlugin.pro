TEMPLATE        = lib
CONFIG         += plugin

HEADERS += \
    acmlvector.h \
    acmlmatrix.h \
    acmlmath.h

SOURCES += \
    acmlvector.cpp \
    acmlmatrix.cpp \
    acmlmath.cpp

INCLUDEPATH = ../../Hipnos

#QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
win32 {

}
unix {
    # multi thread /cpu
    LIBS        += -L/opt/acml5.1.0/gfortran64_mp/lib -lacml_mp
    INCLUDEPATH += /opt/acml5.1.0/gfortran64_mp/include

    # single thread / cpu
    #LIBS        += -L/opt/acml5.1.0/gfortran64/lib -lacml
    #INCLUDEPATH += /opt/acml5.1.0/gfortran64/include
}
TARGET          = $$qtLibraryTarget(HipnosACMLMathPlugin)
DESTDIR         = ../bin


# TODO: as for the cuda plugin try to use Zero Copy mem allocation
# CL_MEM_COPY_HOST_PTR in clCreateBuffer
# see http://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clCreateBuffer.html
# and OpenCL_Best_Practices_Guide.pdf Chapter 3: Memory Optimizations
