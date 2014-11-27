#-------------------------------------------------
#
# Project created by QtCreator 2012-02-14T12:14:35
#
#-------------------------------------------------

QT       += core gui xml #opengl

TARGET = hipnos
TEMPLATE = app

SOURCES += main.cpp\
        mainwindow.cpp \
    common/spectrum.cpp \
    view/irrlichtcontroller.cpp \
    view/cscenenodeanimatorcamerahipnos.cpp \
    view/inputcontroller.cpp \
    design/designerscene.cpp \
    design/designercontroller.cpp \
    analyse/analysiscontroller.cpp \
    common/pipelinedata.cpp \
    common/components/aperturecomponent.cpp \
    common/pipeline.cpp \
    analyse/analysissliderscene.cpp \
    common/pipelinecomponentmanager.cpp \
    design/componentlistwidgetitem.cpp \
    design/componentsceneitem.cpp \
    design/componentstripesceneitem.cpp \
    analyse/slidersceneitem.cpp \
    common/components/thinlenscomponent.cpp \
    common/math/math.cpp \
    analyse/componentpropertywidget.cpp \
    common/components/groupcomponent.cpp \
    common/filesysteminterface.cpp \
    common/xml/customcomponenthandler.cpp \
    common/xml/scenehandler.cpp \
    common/pipelinecomponent.cpp \
    common/widgets/gradienteditor.cpp \
    common/widgets/qtcolortriangle.cpp \
    analyse/vtkviewpropertywidget.cpp \
    analyse/analysiscomponentsceneitem.cpp \
    common/xml/hipnosxmlhandler.cpp \
    analyse/chartsceneitem.cpp \
    analyse/vtkpipeline.cpp \
    common/widgets/mathplugindialog.cpp \
    common/pipelineconnection.cpp \
    common/widgets/ballontip.cpp \
    common/widgets/infobutton.cpp \
    common/widgets/spinbox.cpp \
    common/components/propagationcomponent.cpp \
    analyse/exporttocsvdialog.cpp \
    common/components/gaussianbeamsourcecomponent.cpp \
    common/components/csvsourcecomponent.cpp \
    common/widgets/spectrumdialog.cpp \
    common/widgets/qcustomplot.cpp \
    common/widgets/qxtspanslider.cpp \
    common/components/scriptedcomponent.cpp

HEADERS  += mainwindow.h \
    view/irrlichtcontroller.h \
    view/inputcontroller.h \
    view/cscenenodeanimatorcamerahipnos.h \
    design/designerscene.h \
    design/designercontroller.h \
    analyse/analysiscontroller.h \
    common/pipelinedata.h \
    common/pipelinecomponent.h \
    common/components/aperturecomponent.h \
    common/pipeline.h \
    analyse/analysissliderscene.h \
    common/pipelinecomponentmanager.h \
    design/componentlistwidgetitem.h \
    design/componentsceneitem.h \
    design/componentstripesceneitem.h \
    analyse/slidersceneitem.h \
    common/components/thinlenscomponent.h \
    common/math/math.h \
    analyse/componentpropertywidget.h \
    common/components/groupcomponent.h \
    common/filesysteminterface.h \
    common/xml/customcomponenthandler.h \
    common/xml/scenehandler.h \
    common/statusmessage.h \
    analyse/propertychangedhandler.h \
    common/math/mathplugin.h \
    common/math/matrix.h\
    common/math/vector.h \
    common/widgets/gradienteditor.h \
    common/widgets/qtcolortriangle.h \
    analyse/vtkviewpropertywidget.h \
    analyse/analysiscomponentsceneitem.h \
    common/xml/hipnosxmlhandler.h \
    analyse/chartsceneitem.h \
    analyse/vtkpipeline.h \
    common/widgets/mathplugindialog.h \
    common/hipnossettings.h \
    common/pipelineconnection.h \
    common/threadsafequeue.h \
    common/widgets/ballontip.h \
    common/widgets/infobutton.h \
    common/widgets/spinbox.h \
    common/components/propagationcomponent.h \
    analyse/exporttocsvdialog.h \
    common/components/gaussianbeamsourcecomponent.h \
    common/components/csvsourcecomponent.h \
    common/spectrum.h \
    common/widgets/spectrumdialog.h \
    common/widgets/qcustomplot.h \
    common/widgets/qxtspanslider.h \
    common/widgets/qxtspanslider_p.h \
    common/components/scriptedcomponent.h

FORMS    += mainwindow.ui

DESTDIR  = ../bin

win32 {
    #INCLUDEPATH = ../include/irrlicht/1.7.3
    INCLUDEPATH += ../include/vtk/5.10.0

    #LIBS        += -L$$PWD/../lib/irrlicht/1.7.3
    LIBS        += -L$$PWD/../lib/vtk/5.10.0

    WIN_PWD = $$replace(PWD,/,\\)
    WIN_DEST = $$replace(DESTDIR,/,\\)
    debug {
        QMAKE_PRE_LINK = copy /Y $$WIN_PWD\\..\\dll\\vtk\\5.10.0\\*.dll $$WIN_DEST && \
                         copy /Y $$(QTDIR)\\lib\\QtCored4.dll $$WIN_DEST && \
                         copy /Y $$(QTDIR)\\lib\\QtGuid4.dll $$WIN_DEST && \
                         copy /Y $$(QTDIR)\\lib\\QtNetworkd4.dll $$WIN_DEST && \
                         copy /Y $$(QTDIR)\\lib\\QtSqld4.dll $$WIN_DEST && \
                         copy /Y $$(QTDIR)\\lib\\QtWebKitd4.dll $$WIN_DEST && \
                         copy /Y $$(QTDIR)\\lib\\QtXmld4.dll $$WIN_DEST && \
                         (IF not exist $$WIN_DEST\\imageformats (mkdir $$WIN_DEST\\imageformats)) && \
                         copy /Y $$(QTDIR)\\plugins\\imageformats\\qgifd4.dll $$WIN_DEST\\imageformats
    }
    release {
        QMAKE_PRE_LINK = copy /Y $$WIN_PWD\\..\\dll\\vtk\\5.10.0\\*.dll $$WIN_DEST && \
                         copy /Y $$(QTDIR)\\lib\\QtCore4.dll $$WIN_DEST && \
                         copy /Y $$(QTDIR)\\lib\\QtGui4.dll $$WIN_DEST && \
                         copy /Y $$(QTDIR)\\lib\\QtNetwork4.dll $$WIN_DEST && \
                         copy /Y $$(QTDIR)\\lib\\QtSql4.dll $$WIN_DEST && \
                         copy /Y $$(QTDIR)\\lib\\QtWebKit4.dll $$WIN_DEST && \
                         copy /Y $$(QTDIR)\\lib\\QtXml4.dll $$WIN_DEST && \
                         (IF not exist $$WIN_DEST\\imageformats (mkdir $$WIN_DEST\\imageformats)) && \
                         copy /Y $$(QTDIR)\\plugins\\imageformats\\qgif4.dll $$WIN_DEST\\imageformats
    }
    #copy /Y $$WIN_PWD\\..\\dll\\irrlicht\\1.7.3\\Irrlicht.dll $$WIN_DEST && \

}

unix {
    #INCLUDEPATH = /usr/include/irrlicht
    INCLUDEPATH += /usr/local/include/vtk-5.10
    INCLUDEPATH += /usr/include/vtk-5.8
    LIBS        += -L/usr/local/lib/vtk-5.10
}

#LIBS        += -lIrrlicht
LIBS        += -lvtkCommon -lvtksys -lQVTK -lvtkViews -lvtkWidgets -lvtkInfovis -lvtkRendering -lvtkGraphics -lvtkImaging -lvtkIO -lvtkFiltering -lvtkDICOMParser -lvtkalglib -lvtkverdict -lvtkmetaio -lvtkexoIIc -lvtkftgl -lvtkHybrid

#INCLUDEPATH += ../QIrrlichtWidget
#LIBS        += -L..\bin\win -lqirrlichtwidgetplugin
#LIBS        += ..\bin\win\qirrlichtwidgetplugin.lib

RESOURCES += \
    resources.qrc

OTHER_FILES += \
    icons/*.png \
    icons/components/* \
    icons/animations/* \
    qdoc.qdocconf
