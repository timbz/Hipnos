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

#include "filesysteminterface.h"
#include "xml/customcomponenthandler.h"
#include "xml/scenehandler.h"
#include "design/componentstripesceneitem.h"
#include "design/componentsceneitem.h"
#include "hipnossettings.h"

#include <QXmlSimpleReader>
#include <QFileDialog>
#include <QPluginLoader>
#include <QDebug>

FileSystemInterface::FileSystemInterface()
{
    qDebug() << "Initializing FileSystemInterface";
    if(!QDir::home().exists(HIPNOS_HOME_DIR_NAME))
    {
        QDir::home().mkdir(HIPNOS_HOME_DIR_NAME);
    }
    hipnosHome = QDir::home();
    hipnosHome.cd(HIPNOS_HOME_DIR_NAME);

    pluginDir  = QDir::current();
    pluginDir.cdUp();
    pluginDir.cd("plugins");
    pluginDir.cd("bin");

    logFilePath = hipnosHome.path() + QDir::separator() + "hipnos.log";
    customComponentsFilePath = hipnosHome.path() + QDir::separator() + "customComponents.xml";
    hipnosFileExtension = ".hipnos";
}

QString FileSystemInterface::getLoggingPath()
{
    return logFilePath;
}

QDir FileSystemInterface::getHipnosHome()
{
    return hipnosHome;
}

void FileSystemInterface::loadCustomComponents()
{
    QFile file(customComponentsFilePath);
    if(!file.open(QFile::ReadOnly | QFile::Text))
    {
        qWarning() << "Could not read file " + customComponentsFilePath;
    }
    QXmlInputSource source(&file);
    CustomComponentHandler handler;
    QXmlSimpleReader xmlReader;
    xmlReader.setContentHandler(&handler);
    xmlReader.setErrorHandler(&handler);
    if(!xmlReader.parse(&source))
    {
        qWarning() << "Could not read custom components from file " + customComponentsFilePath;
    }
    file.close();
}

QString FileSystemInterface::getIndent(uint indent)
{
    QString tmp;
    for(uint i = 0; i < indent; i++)
        tmp += "    ";
    return tmp;
}

QString FileSystemInterface::componentToXML(PipelineComponent *c, uint indent)
{
    QString xml;
    xml += getIndent(indent) + "<Component type=\"" + c->getType() + "\" name=\"" + c->getName() + "\">\n";
    foreach(PipelineComponent::Property p, c->getProperties())
    {
        xml += getIndent(indent+1) + "<Property name=\"" + p.getName() + "\" value=\"" + p.currentValueToString() + "\" />\n";
    }
    xml += getIndent(indent) + "</Component>\n";
    return xml;
}

void FileSystemInterface::saveCustomComponents(QHash<QString, GroupComponent *> components)
{
    QFile file(customComponentsFilePath);
    if(!file.open(QFile::WriteOnly | QFile::Truncate))
    {
        qWarning() << "Could not open file " + customComponentsFilePath;
        return;
    }
    QTextStream out(&file);
    out.setCodec("UTF-8");
    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    out << "<GroupComponents>\n";

    QList<QString> savedComponentKeys;

    int componentSize = components.size();
    for(int i = 0; i < componentSize; i++)
    {
        GroupComponent* current = 0;
        foreach(GroupComponent* g, components)
        {
            // get the dependencies
            QList<QString> dep = g->getDependencyKeys();
            // remove already stored dependencies
            foreach(QString s, savedComponentKeys)
                dep.removeAll(s);
            if(dep.isEmpty())
            {
                current = g;
                break;
            }
        }
        if(current == 0)
        {
            qCritical() << "An error occurred while saving custom components: Could not resolve dependency graph";
            break;
        }
        savedComponentKeys << current->getType();
        components.remove(current->getType());

        // save the component
        QColor color = current->getColor();
        QString colorString = QString::number(color.red()) + "," + QString::number(color.green()) + "," + QString::number(color.blue()) + "," + QString::number(color.alpha());
        out << "    <GroupComponent name=\"" << current->getType() << "\" color=\"" << colorString << "\">\n";
        foreach(PipelineComponent* c, current->getComponents())
        {
            out << componentToXML(c, 2);
        }
        out << "    </GroupComponent>\n";
    }
    if(!components.empty())
    {
        qCritical() << "An error occurred while saving custom components.";
    }
    out << "</GroupComponents>\n";
    file.close();
}

void FileSystemInterface::saveDesignerScene(DesignerScene *scene)
{
    QString filePath = QFileDialog::getSaveFileName(0, "Save...", hipnosHome.path(), "Hipnos Files (*" + hipnosFileExtension + ")");
    if(!filePath.isEmpty() && !filePath.isNull())
    {
        if(!filePath.endsWith(hipnosFileExtension))
            filePath += hipnosFileExtension;
        QFile file(filePath);
        if(!file.open(QFile::WriteOnly | QFile::Truncate))
        {
            qWarning() << "Could not open file " + filePath;
            return;
        }
        QTextStream out(&file);
        out.setCodec("UTF-8");
        out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        out << "<Scene>\n";
        out << "    <Stripe slots=\"" << QString::number(scene->getStripe()->getComponetSceneItems().size()) << "\" >\n";
        foreach(ComponentSceneItem* item, scene->getStripe()->getComponetSceneItems())
        {
            out << "        <Slot>\n";
            if(item)
            {
                out << componentToXML(item->getComponent(), 3);
            }
            out << "        </Slot>\n";
        }
        out << "    </Stripe>\n";
        out << "</Scene>\n";
        file.close();
    }
}


void FileSystemInterface::loadDesignerScene(DesignerScene *scene)
{
    QString filePath = QFileDialog::getOpenFileName(0, "Open...", hipnosHome.path(), "Hipnos Files (*" + hipnosFileExtension + ")");
    if(!filePath.isEmpty() && !filePath.isNull())
    {
        if(!filePath.endsWith(hipnosFileExtension))
            filePath += hipnosFileExtension;
        QFile file(filePath);
        if(!file.open(QFile::ReadOnly | QFile::Text))
        {
            qWarning() << "Could not read from file " + filePath;
            return;
        }
        QXmlInputSource *source = new QXmlInputSource(&file);
        SceneHandler* handler = new SceneHandler(scene);
        QXmlSimpleReader xmlReader;
        xmlReader.setContentHandler(handler);
        xmlReader.setErrorHandler(handler);
        if(!xmlReader.parse(source))
        {
            qWarning() << "Could not load pipeline from file " + filePath;
        }
        file.close();
    }
}

QList<MathPlugin*> FileSystemInterface::loadMathPlugins()
{
    qDebug() << "Loading Math Plugins ...";
    QList<MathPlugin*> plugins;
    foreach (QString fileName, pluginDir.entryList(QDir::Files))
    {
//        if(!fileName.contains("CBlas"))
//            continue;
        QPluginLoader pluginLoader(pluginDir.absoluteFilePath(fileName));
        QObject *plugin = pluginLoader.instance();
        if(plugin) {
            MathPlugin* mathPlugin  = qobject_cast<MathPlugin *>(plugin);
            if(mathPlugin)
            {
                if(mathPlugin->isPlatformSupported())
                {
                    qDebug() << mathPlugin->getName() << "loaded.";
                    plugins << mathPlugin;
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
        }
    }
    return plugins;
}
