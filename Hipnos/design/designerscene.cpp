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

#include "designerscene.h"
#include "designercontroller.h"
#include "common/pipelinecomponentmanager.h"
#include <QApplication>
#include <QInputDialog>
#include <QColorDialog>
#include <QDebug>

DesignerScene::DesignerScene(DesignerController* c) :
    QGraphicsScene()
{
    isDragging = false;
    draggedItem = 0;
    graphicView = c->getGraphicView();
    graphicView->setAcceptDrops(true);
    graphicView->setScene(this);
    stripe = new ComponentStripeSceneItem();
    addItem(stripe);
    connect(stripe, SIGNAL(stripeModified()), c, SLOT(setPipelineModified()));
}

void DesignerScene::mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    isDragging = false;
    QGraphicsScene::mousePressEvent(mouseEvent);
}

void DesignerScene::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    if (QLineF(event->screenPos(), event->buttonDownScreenPos(Qt::LeftButton)).length() >= QApplication::startDragDistance())
    {
        isDragging = true;
    }
    QGraphicsScene::mouseMoveEvent(event);
}

void DesignerScene::mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    QGraphicsScene::mouseReleaseEvent(mouseEvent);
}

ComponentSceneItem* DesignerScene::getDraggedItem()
{
    return draggedItem;
}

void DesignerScene::setDraggedItem(ComponentSceneItem* item)
{
    draggedItem = item;
}

Pipeline* DesignerScene::getPipeline()
{
    PipelineComponent* first = stripe->buildComponetStripe();
    if(first)
    {
        Pipeline* p = new Pipeline();
        p->setFirstComponent(first);
        return p;
    }
    else
        return 0;
}

ComponentStripeSceneItem* DesignerScene::getStripe()
{
    return stripe;
}

void DesignerScene::reset()
{
    foreach(ComponentSceneItem* item, stripe->getComponetSceneItems())
    {
        if(item)
        {
            stripe->removeItem(item, false);
            delete item;
        }
    }
    stripe->getComponetSceneItems().resize(1);
    update();
}

void DesignerScene::deleteSelectedItems()
{
    QList<QGraphicsItem*> selection = selectedItems();
    for(int i = 0; i < selection.size(); i++)
    {
        ComponentSceneItem* item = dynamic_cast<ComponentSceneItem*>(selection.at(i));
        if(item)
        {
            stripe->removeItem(item);
        }
    }
}

GroupComponent* DesignerScene::groupSelectedItems()
{
    QList<ComponentSceneItem*> selectedSceneItems;
    foreach(QGraphicsItem* i, selectedItems())
    {
        ComponentSceneItem* item = dynamic_cast<ComponentSceneItem*>(i);
        if(item)
        {
            selectedSceneItems.push_back(item);
        }
    }
    if(selectedSceneItems.size() <= 0)
    {
        qDebug() << "No item selected for grouping";
        return 0;
    }
    qSort(selectedSceneItems.begin(), selectedSceneItems.end(), ComponentSceneItem::LessThan());

    // check if item chain has holes
    for(int i = 1; i < selectedSceneItems.size(); i++)
    {
        if(QLineF(selectedSceneItems.at(i)->pos(), selectedSceneItems.at(i-1)->pos()).length() > stripe->getSlotDistance())
        {
            qDebug() << "Selected items are not consecutive";
            return 0;
        }
    }

    // cal slotID
    int slotId = stripe->getSlotIdByPosition(selectedSceneItems.front()->pos());
    if(slotId < 0)
    {
        qWarning() << "Error calculating slot id";
        return 0;
    }

    // test if we can build a linear pipeline form the selected items
    for(int i = 1; i < selectedSceneItems.size(); i++)
    {
        if(selectedSceneItems.at(i-1)->getComponent()->getNumberOfOutputConnections() != 1
                || selectedSceneItems.at(i)->getComponent()->getNumberOfInputConnections() != 1)
        {
            qDebug() << "Cannot connect seleted items";
            return 0;
        }
    }

    // build component list and test if the selected componentes are basic components
    QList<PipelineComponent*> componentsToGroup;
    foreach(ComponentSceneItem* item, selectedSceneItems)
    {
        componentsToGroup << item->getComponent()->clone();
    }

    // build component group
    bool ok;
    QString name;
    do
    {
        name = QInputDialog::getText(graphicView, tr("QInputDialog::getText()"),
                                         tr("Group name:"), QLineEdit::Normal,
                                         "Group name", &ok);
        if (!ok)
        {
            return 0; // user clicked cancel
        }
    }
    while(!PipelineComponentManager::getInstance().isTypeNameValid(name));

    QColor color = QColorDialog::getColor(QColor(0,0,100));
    GroupComponent* group = GroupComponent::create(name, componentsToGroup, color);
    if(group == 0)
    {
        qWarning() << "Could not create component group";
        return 0;
    }

    // if all went well delete the grouped items and add the group
    foreach(ComponentSceneItem* item, selectedSceneItems)
    {
        stripe->removeItem(item, false);
        removeItem(item);
        delete item;
    }
    stripe->addComponentToSlot(group->clone(), slotId);

    return group;
}

void DesignerScene::ungroupSelectedItems()
{
    // get all group items
    QList<ComponentSceneItem*> selectedGroupSceneItems;
    foreach(QGraphicsItem* i, selectedItems())
    {
        ComponentSceneItem* item = dynamic_cast<ComponentSceneItem*>(i);
        if(item)
        {
            GroupComponent* group = dynamic_cast<GroupComponent*>(item->getComponent());
            if(group)
            {
                selectedGroupSceneItems << item;
                // calc slotID
                int slotId = stripe->getSlotIdByPosition(item->pos());
                if(slotId < 0)
                {
                    qCritical() << "Error calculating slot id";
                    continue;
                }
                // ungroup
                int i = 0;
                foreach(PipelineComponent* c, group->ungroup())
                {
                    if(i == 0)
                    {
                        // we add the first component to the slot of the group
                        stripe->addComponentToSlot(c, slotId);
                    }
                    else
                    {
                        stripe->insertSlotAfterSlot(slotId+i-1);
                        stripe->addComponentToSlot(c, slotId+i);
                    }
                    i++;
                }
            }
        }
    }
    // if all went well delete the groups
    foreach(ComponentSceneItem* item, selectedGroupSceneItems)
    {
        stripe->removeItem(item, false);
        removeItem(item);
        delete item;
    }
}
