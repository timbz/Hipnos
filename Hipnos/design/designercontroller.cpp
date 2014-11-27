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

#include "designercontroller.h"
#include "common/pipelinecomponentmanager.h"
#include "common/filesysteminterface.h"
#include "componentlistwidgetitem.h"
#include <QDebug>

DesignerController::DesignerController(Ui::MainWindow *ui) :
    QObject()
{
    zoomFactor = 1.2;
    currentPipeline = 0;
    pipelineModified = true;
    view = ui->graphicsView;
    scene = new DesignerScene(this);
    baseComponentList = ui->componentListWidget;
    customComponentList = ui->customComponentListWidget;

    QList<PipelineComponent*> basicComponents = PipelineComponentManager::getInstance().getBasicComponents().values();
    qSort(basicComponents.begin(), basicComponents.end(), PipelineComponent::LessThan());
    foreach(PipelineComponent* c, basicComponents)
    {
        baseComponentList->addItem(new ComponentListWidgetItem(c));
    }
    QList<GroupComponent*> customComponents = PipelineComponentManager::getInstance().getCustomComponents().values();
    qSort(customComponents.begin(), customComponents.end(), PipelineComponent::LessThan());
    foreach(GroupComponent* c, customComponents)
    {
        customComponentList->addItem(new ComponentListWidgetItem(c));
    }
    // zoom out one step
    zoomOut();
}

DesignerController::~DesignerController()
{
    delete scene;
}

void DesignerController::setPipelineModified()
{
    pipelineModified = true;
}

bool DesignerController::getPipelineModified()
{
    return pipelineModified;
}

void DesignerController::zoomOut()
{
    view->scale(1.0/zoomFactor, 1.0/zoomFactor);
}

void DesignerController::zoomIn()
{
    view->scale(zoomFactor, zoomFactor);
}

void DesignerController::deleteSelectedItems()
{
    scene->deleteSelectedItems();
}

void DesignerController::groupSelectedItems()
{
    GroupComponent* group = scene->groupSelectedItems();
    if(group)
    {
        // Save the component
        PipelineComponentManager::getInstance().addCustomComponent(group);
        PipelineComponentManager::getInstance().saveCustomComponents();
        customComponentList->addItem(new ComponentListWidgetItem(group));
    }
}

void DesignerController::ungroupSelectedItems()
{
    scene->ungroupSelectedItems();
}

void DesignerController::deleteCustomComponent()
{
    ComponentListWidgetItem* selectedItem = dynamic_cast<ComponentListWidgetItem*>(customComponentList->currentItem());
    if(selectedItem)
    {
        if(PipelineComponentManager::getInstance().deleteCustomComponent(
                    selectedItem->getComponentPrototype()->getType()))
        {
            PipelineComponentManager::getInstance().saveCustomComponents();
            customComponentList->removeItemWidget(selectedItem);
            delete selectedItem;
        }
    }
}

QGraphicsView* DesignerController::getGraphicView()
{
    return view;
}

Pipeline* DesignerController::getPipeline()
{
    if(pipelineModified)
    {
        if(currentPipeline)
        {
            delete currentPipeline;
        }
        currentPipeline = scene->getPipeline();
        pipelineModified = false;
    }
    return currentPipeline;
}

void DesignerController::save()
{
    FileSystemInterface::getInstance().saveDesignerScene(scene);
}

void DesignerController::load()
{
    pipelineModified = true;
    scene->reset();
    FileSystemInterface::getInstance().loadDesignerScene(scene);
}
