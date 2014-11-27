QT       += testlib

QT       -= gui

TARGET = tst_mathpluginstest
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

DESTDIR = ../bin

INCLUDEPATH = ../../Hipnos

SOURCES += tst_mathpluginstest.cpp
DEFINES += SRCDIR=\\\"$$PWD/\\\"
