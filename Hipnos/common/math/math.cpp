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

#include "math.h"
#include "common/filesysteminterface.h"
#include "common/widgets/mathplugindialog.h"

bool Math::initialised = false;
MathPlugin* Math::instance = 0;
QList<MathPlugin*> Math::mathPlugins = QList<MathPlugin*>();

void Math::init()
{
    if(!initialised)
    {
        mathPlugins = FileSystemInterface::getInstance().loadMathPlugins();
        if(mathPlugins.size() > 0)
        {
            instance = MathPluginDialog::selectPlugin(mathPlugins);
            qDebug() << "Using" << instance->getName();
            initialised = true;
        }
        else
        {
            qFatal("No Math Plugin found");
        }
    }
}
