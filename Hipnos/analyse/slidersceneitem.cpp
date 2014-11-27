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

#include "slidersceneitem.h"
#include <QCursor>

SliderSceneItem::SliderSceneItem(qreal s) :
    QGraphicsItem()
{
    size = s;
    setCursor(Qt::OpenHandCursor);
}

QRectF SliderSceneItem::boundingRect() const
{
    return QRectF(-size/4,-size/2,size/2,size);
}

void SliderSceneItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    painter->setBrush(QBrush(QColor(255,0,0)));
    painter->drawLine(0,-size/2,0,size/2);

    QPointF arrow1[3];
    arrow1[0] = QPointF(0,-size/4);
    arrow1[1] = QPointF(-size/8,-size/2);
    arrow1[2] = QPointF(+size/8,-size/2);
    painter->drawPolygon(arrow1, 3);

    QPointF arrow2[3];
    arrow2[0] = QPointF(0, size/4);
    arrow2[1] = QPointF(-size/8, size/2);
    arrow2[2] = QPointF(+size/8, size/2);
    painter->drawPolygon(arrow2, 3);
}

void SliderSceneItem::draggingStart()
{
    setCursor(Qt::ClosedHandCursor);
}

void SliderSceneItem::draggingEnd()
{
    setCursor(Qt::OpenHandCursor);
}
