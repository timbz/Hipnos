TEMPLATE        = lib
CONFIG         += plugin

HEADERS += \
    cudamath.h \
    cublasmatrix.h \
    cublasvector.h

SOURCES += \
    cudamath.cpp \
    cublasmatrix.cpp \
    cublasvector.cpp

INCLUDEPATH = ../../Hipnos

win32 {
    LIBS        +=  "$$(CUDA_LIB_PATH)\\..\\Win32\\cudart.lib"
    LIBS        +=  "$$(CUDA_LIB_PATH)\\..\\Win32\\cublas.lib"
    INCLUDEPATH += $(CUDA_INC_PATH)
}
unix {
    INCLUDEPATH += /usr/local/cuda/include/
    LIBS        += -L/usr/local/cuda/lib64 -lcublas -lcufft -lcudart
}

TARGET          = $$qtLibraryTarget(HipnosCudaMathPlugin)
DESTDIR         = ../bin


# TUTORIAL: Use NVCC in Qt
# http://cudaspace.wordpress.com/2012/07/05/qt-creator-cuda-linux-review/
#QMAKE_CXXFLAGS_RELEASE = -O3
QMAKE_CXXFLAGS = -fPIC
QMAKE_CFLAGS = -fPIC
# Cuda sources
CUDA_SOURCES += kernels.cu
OTHER_FILES  += $$CUDA_SOURCES
# Path to cuda toolkit install
CUDA_DIR      = /usr/local/cuda
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64     # Note I'm using a 64 bits Operating system
# GPU architecture
CUDA_ARCH     = sm_21
# Here are some NVCC flags I've always used by default.
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math #--ptxas-options=-v

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c -Xcompiler -fpic $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2

cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda

# TODO: Add cuda math plugin with mapped pinned mem for zero copy devices (GPU integrated in CPU)
#   From http://www.drdobbs.com/parallel/cuda-supercomputing-for-the-masses-part/217500110
#   Integrated GPUs share physical memory with the host processor (as opposed to the on-board fast global memory
#   of discrete GPUs). Mapped pinned buffers act as "zero-copy" buffers for many newer (especially integrated
#   graphics processors) because they avoid superfluous copies. When developing code for integrated GPUs,
#   using mapped pinned memory really makes sense.
#   For discrete GPUs, mapped pinned memory is only a performance win in certain cases. Since the memory is not
#   cached by the GPU:
#   - It should be read or written exactly once.
#   - The global loads and stores that read or write the memory must be coalesced to avoid a 2x-7x PCIe
#     performance penalty.
#   - At best, it will only deliver PCIe bandwidth performance, but this can be 2x faster than cudaMemcpy
#     because mapped memory is able exploit the full duplex capability of the PCIe bus by reading and writing
#     at the same time. A call to cudaMemcpy can only move data in one direction at a time (i.e., half duplex).
#
# Example: http://webcache.googleusercontent.com/search?q=cache:ttT7G7HA2BsJ:forums.nvidia.com/index.php%3Fshowtopic%3D166526+cuda+mapped+memory&cd=2&hl=de&ct=clnk&gl=de
#          CUDA_C_Best_Practices_Guide.pdf Chapter: Zero Copy
