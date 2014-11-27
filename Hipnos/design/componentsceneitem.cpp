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

#include "componentsceneitem.h"
#include "designerscene.h"
#include "componentstripesceneitem.h"
#include <QDrag>
#include <QCursor>
#include <QGraphicsSceneMouseEvent>
#include <QApplication>
#include <QMimeData>
#include <QDropEvent>
#include <QMenu>
#include <QDebug>

ComponentSceneItem::ComponentSceneItem(ComponentStripeSceneItem* s, PipelineComponent* c, QRectF b)
    : QGraphicsObject(s)
{
    stripe = s;
    component = c;
    bbox = b;
    isBeingDragged = false;
    setCursor(Qt::OpenHandCursor);
    setFlag(ItemIsSelectable);
    setSelected(false);
}

QRectF ComponentSceneItem::boundingRect() const
{
    return bbox;
}

void ComponentSceneItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
           QWidget *widget)
{
    setToolTip(component->getName());
    // gray out if we are draggin this item
    if(isBeingDragged)
        painter->setOpacity(0.3);
    painter->drawPixmap(0, 0, bbox.width(),bbox.height(),
                        component->getIcon().pixmap(bbox.width(),bbox.height()));
    if(isSelected())
    {
        painter->setBrush(QColor(200,200,200, 100));
        painter->drawRect(0,0,bbox.width()-1,bbox.height()-1);
    }
}

void ComponentSceneItem::mousePressEvent(QGraphicsSceneMouseEvent * ev)
 {
    QGraphicsObject::mousePressEvent(ev);
 }

void ComponentSceneItem::setIsBeingDragged(bool b)
{
    isBeingDragged = b;
}

PipelineComponent* ComponentSceneItem::getComponent()
{
    return component;
}

void ComponentSceneItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
     if (QLineF(event->screenPos(), event->buttonDownScreenPos(Qt::LeftButton)).length() >= QApplication::startDragDistance()) {
         startDrag();
     }
     QGraphicsObject::mouseMoveEvent(event);
}

void ComponentSceneItem::startDrag()
{
    QDrag *drag = new QDrag(scene()->views().first());
    QMimeData *mime = new QMimeData;
    drag->setMimeData(mime);

    drag->setPixmap(component->getIcon().pixmap(50));
    drag->setHotSpot(QPoint(25, 25));
    DesignerScene* s = dynamic_cast<DesignerScene*>(scene());
    if(s) s->setDraggedItem(this);
    isBeingDragged = true;
    // redraw
    update();
    drag->exec();
    isBeingDragged = false;
    // redraw
    update();
}

void ComponentSceneItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    QGraphicsObject::mouseReleaseEvent(event);
}

void ComponentSceneItem::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
    QMenu menu;
    QAction *removeAction = menu.addAction("Remove");
    QAction *selectedAction = menu.exec(event->screenPos());
    if(selectedAction == removeAction)
    {
        if(stripe)
        {
            stripe->removeItem(this);
        }
    }
}

ComponentSceneItem::~ComponentSceneItem()
{
    delete component;
}
