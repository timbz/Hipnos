/****************************************************************************
** 
** This file is part of HiPNOS.
** 
** Copyright 2012 Helmholtz-Zentrum Dresden-Rossendorf
** 
** HiPNOS is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** HiPNOS is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with HiPNOS.  If not, see <http://www.gnu.org/licenses/>.
** 
****************************************************************************/

#include "mainwindow.h"
#include "common/filesysteminterface.h"
#include "common/pipelinecomponentmanager.h"
#include "common/math/math.h"

#include <QApplication>
#include <fstream>
#include <QTime>
#include <iostream>
#include <cstdio>

#include "common/widgets/spectrumdialog.h"

std::ofstream logfile; 

/**
 * @brief
 *
 * @param type
 * @param msg
 */
void SimpleLoggingHandler(QtMsgType type, const char *msg)
{
    std::string time = QTime::currentTime().toString().toStdString();
    switch (type)
    {
        case QtDebugMsg:
            logfile << time << " Debug: " << msg << "\n";
            std::cout << time << " Debug: " << msg << std::endl;
            break;
        case QtWarningMsg:
            logfile << time << " Warning: " << msg << "\n";
            std::cout << time << " Warning: " << msg << std::endl;
            break;
        case QtCriticalMsg:
            logfile << time << " Critical: " << msg << "\n";
            std::cout << time << " Critical: " << msg << std::endl;
            logfile.flush();
            break;
        case QtFatalMsg:
            logfile << time <<  " Fatal: " << msg << "\n";
            std::cout << time << " Fatal: " << msg << std::endl;
            logfile.flush();
            QApplication::exit(0);
    }
}

/**
 * @brief
 *
 * @param argc
 * @param argv[]
 * @return int
 */
int main(int argc, char *argv[])
{
    // Register some custom types
    qRegisterMetaType<Spectrum>();

    // Logging
    std::remove(FileSystemInterface::getInstance().getLoggingPath().toLocal8Bit().constData());
    logfile.open(FileSystemInterface::getInstance().getLoggingPath().toLocal8Bit().constData(), ios::app);
    qInstallMsgHandler(SimpleLoggingHandler);
    qDebug() << "Logging initialized";

    QApplication a(argc, argv);

    // Init
    Math::init();
    PipelineComponentManager::getInstance().loadCustomComponents();

    MainWindow w;

    w.showMaximized();
    return a.exec();
}

/**
\mainpage Mainpage documentation

This application is a simulation software for high performance lasers.
The graphical interface privides the user with the ability to string together
optical components to build simple beam lines.

The propagation of light through the system can be simulated with the gaussian
beam model and the fourier optics model. Intensive mathematical operations
are sourced out from the main application into plugins.

As a consequence of this design it is possible to implement different versions
of the plugin interface tailored towards specific system configurations.

@image html arch-small.png
@image latex arch-small.png "Hipnos - Architecture"

*/
