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

#include "componentstripesceneitem.h"

#include <QListWidget>
#include <QGraphicsScene>
#include <QGraphicsView>
#include "componentlistwidgetitem.h"
#include "designerscene.h"
#include "componentsceneitem.h"
#include "common/pipeline.h"

#include <QDebug>
ComponentStripeSceneItem::ComponentStripeSceneItem() :
    QGraphicsObject()
{
    setAcceptDrops(true);
    slotWidth = 100;
    slotHeight = 100;
    padding = 10;
    componentSlots.resize(1);
    background = new QPixmap(":/icons/designerstripe.png");
}

ComponentStripeSceneItem::~ComponentStripeSceneItem()
{
    delete background;
}

QRectF ComponentStripeSceneItem::boundingRect() const
{
    qreal marginx = slotWidth/3;
    qreal marginy = slotHeight/3;
    return QRectF(-marginx, -marginy, (slotWidth+padding)*componentSlots.size() + 2*marginx, slotHeight + 2*marginy);
}

void ComponentStripeSceneItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    for(int i = 0; i < componentSlots.size(); i++)
    {
        painter->drawPixmap(QRect(i*(slotWidth+padding), -50, slotWidth, slotHeight+100), *background);
        if(componentSlots[i])
        {
            componentSlots[i]->setPos(QPointF(i*(slotWidth+padding), 0));
        }
    }
}

// TODO: Revrite drag n drop code
void ComponentStripeSceneItem::dropEvent(QGraphicsSceneDragDropEvent *event)
{
    int id = getSlotIdByPosition(event->scenePos());
    if(id < 0 || id >= componentSlots.size()) return;

    // try to get the component item
    ComponentSceneItem* item = 0;
    QListWidget* listWidget = qobject_cast<QListWidget*>(event->source());
    // TODO: find a way to do this switch better
    if(listWidget)
    {
        // the drag comes from the ListWidget, we create a new item
        ComponentListWidgetItem* selectedItem = dynamic_cast<ComponentListWidgetItem*>(listWidget->currentItem());
        if(selectedItem)
        {
            item = new ComponentSceneItem(this, selectedItem->getComponentPrototype()->clone(), QRectF(0,0,slotWidth,slotHeight));
        }
    }
    else if(event->source() == scene()->views().first())
    {
        // the drag comes from an other slot
        DesignerScene* s = dynamic_cast<DesignerScene*>(scene());
        if(s)
        {
            item = s->getDraggedItem();
            if(item)
                removeItem(item, false);
            // unset the dragged item in the scene
            s->setDraggedItem(0);
        }
    }

    if(item)
    {
        // set the item in the vector
        addComponentItemToSlot(item, id);
        item->setIsBeingDragged(false);
    }
}

void ComponentStripeSceneItem::autoResizeStripe()
{
    // see if we can delete some emty slots at the end of the stripe
    int countEmptySlots = 0;
    for(int i = componentSlots.size()-1; i >= 0; i--)
    {
        if(componentSlots[i] == NULL) countEmptySlots++;
        else break;
    }
    if(countEmptySlots > 1)
    {
        componentSlots.resize(componentSlots.size()-countEmptySlots+1);
        prepareGeometryChange();
    }

    // if we filled the last slot we add an other slot to the end of the stripe
    if(componentSlots[componentSlots.size()-1])
    {
        componentSlots.resize(componentSlots.size()+1);
        prepareGeometryChange();
    }
}

void ComponentStripeSceneItem::dragMoveEvent(QGraphicsSceneDragDropEvent *event)
{
    int id = getSlotIdByPosition(event->scenePos());
    if(id < 0 || id >= componentSlots.size())
    {
        event->ignore();
        return;
    }
    if(componentSlots[id])
        event->ignore();
    else
        event->setAccepted(true);
}

void ComponentStripeSceneItem::removeItem(ComponentSceneItem *item, bool deleteItem)
{
    for(int i = 0; i < componentSlots.size(); i++)
    {
        if(item == componentSlots[i])
        {
            componentSlots[i] = 0;
            if(deleteItem)
            {
                autoResizeStripe();
                //NOTE: first redraw, then remove
                scene()->update();
                scene()->removeItem(item);
                item->deleteLater();
            }
            break;
        }
    }
    emit stripeModified();
}

int ComponentStripeSceneItem::getSlotIdByPosition(QPointF p)
{
    qreal xoffset = (p - pos()).x();
    int id = (int)xoffset/(slotWidth+padding);
    if(id >= componentSlots.size() || id < 0)
        return -1;
    else
        return id;
}

void ComponentStripeSceneItem::addComponentToSlot(PipelineComponent *c, int slot)
{
    addComponentItemToSlot(new ComponentSceneItem(this, c, QRectF(0, 0, slotWidth, slotHeight)), slot);
}

void ComponentStripeSceneItem::addComponentItemToSlot(ComponentSceneItem *item, int slot)
{
    componentSlots[slot] = item;
    autoResizeStripe();
    emit stripeModified();
    scene()->update();
}

bool ComponentStripeSceneItem::validateStripe()
{
    // emty, return not valid
    if(componentSlots.size() <= 1)
        return false;
    for(int i = 0; i < componentSlots.size()-1; i++)
    {
        PipelineComponent* curr = 0;
        if(componentSlots[i])
            curr = componentSlots[i]->getComponent();

        if(curr == 0 && i != componentSlots.size()-1)
        {
            qDebug() << "only the last slot can be empty";
            return false;
        }
        if(i == 0)
        {
            if(curr->getNumberOfInputConnections() != 0 || curr->getNumberOfOutputConnections() != 1)
            {
                qDebug() << "the first element must be a source (input connections = 0)";
                return false;
            }
        }
        else
        {
            if(curr->getNumberOfInputConnections() != 1 || curr->getNumberOfOutputConnections() != 1)
            {
                qDebug() << "element not supported (input connection = output connection = 1)";
                return false;
            }
        }
    }
    return true;
}


PipelineComponent* ComponentStripeSceneItem::buildComponetStripe()
{
    if(!validateStripe())
        return 0;
    // NOTE: the last slot is always empty
    for(int i = 0; i < componentSlots.size()-2; i++)
    {
        Pipeline::pipelineConnection(componentSlots[i]->getComponent(), componentSlots[i+1]->getComponent());
    }
    // we add a PipelineSink for each output in the last component
    PipelineComponent* last = componentSlots[componentSlots.size()-2]->getComponent();
    for(int i = 0; i < last->getNumberOfOutputConnections(); i++)
    {
        PipelineSink* sink = new PipelineSink(last);
        last->setOutputConnection(sink, i);
    }
    return componentSlots[0]->getComponent();
}

QVector<ComponentSceneItem*>& ComponentStripeSceneItem::getComponetSceneItems()
{
    return componentSlots;
}

int ComponentStripeSceneItem::getSlotHeight()
{
    return slotHeight;
}

int ComponentStripeSceneItem::getSlotDistance()
{
    return slotWidth + padding;
}

void ComponentStripeSceneItem::insertSlotAfterSlot(int i)
{
    componentSlots.resize(componentSlots.size()+1);
    for(int j = componentSlots.size()-2; j > i; j--)
    {
        if(componentSlots[j])
        {
            componentSlots[j+1] = componentSlots[j];
            componentSlots[j] = 0;
        }
    }
    emit stripeModified();
    scene()->update();
}
