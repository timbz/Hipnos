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

#include "common/math/math.h"
#include <QDir>
#include <QPluginLoader>

bool Math::initialised = false;
MathPlugin* Math::instance = 0;
QList<MathPlugin*> Math::mathPlugins = QList<MathPlugin*>();

void Math::init()
{
    if(!initialised)
    {
        qDebug() << "Loading Math Plugins ...";
        QDir pluginDir = QDir::current();
        pluginDir.cdUp();
        pluginDir.cdUp();
        pluginDir.cd("plugins");
        pluginDir.cd("bin");
        foreach (QString fileName, pluginDir.entryList(QDir::Files))
        {
            QPluginLoader pluginLoader(pluginDir.absoluteFilePath(fileName));
            QObject *plugin = pluginLoader.instance();
            if(plugin)
            {
                MathPlugin* mathPlugin = qobject_cast<MathPlugin *>(plugin);
                if(mathPlugin)
                {
                    if(mathPlugin->isPlatformSupported()
                            && mathPlugin->getDataType() == MathPlugin::DT_DOUBLE_PRECISION)
                    {
                        qDebug() << mathPlugin->getName() << "loaded.";
                        mathPlugins.push_back(mathPlugin);
                    }
                    else
                    {
                        qDebug() << mathPlugin->getName() << "not supported.";
                        delete mathPlugin;
                    }
                }
            }
            else
            {
                qWarning() << pluginLoader.errorString();
                pluginLoader.unload();
            }
        }
        qDebug() << "Found" << mathPlugins.size() << "Math Plugins";

        if(mathPlugins.size() > 0)
        {
            uint performanceHint = 0;
            foreach(MathPlugin* p, mathPlugins)
            {
                if(p->getPerformanceHint() >= performanceHint)
                {
                    performanceHint = p->getPerformanceHint();
                    instance = p;
                }
            }
            qDebug() << "Using" << instance->getName();
            initialised = true;
        }
        else
        {
            qFatal("No Math Plugin found");
        }
    }
}
