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

#include "analysiscomponentsceneitem.h"
#include <QDebug>
#include <QCursor>

AnalysisComponentSceneItem::AnalysisComponentSceneItem(PipelineComponent *c, qreal s) :
    QGraphicsItem()
{
    component = c;
    size = s;
    border = 3;
    setCursor(Qt::PointingHandCursor);
    setAcceptedMouseButtons(Qt::LeftButton);
    setFlag(ItemIsSelectable);
}

QRectF AnalysisComponentSceneItem::boundingRect() const
{
    return QRectF(-size/2, -size/2, size, size);
}

void AnalysisComponentSceneItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    setToolTip(component->getName());
    QSizeF actualIconSize = component->getIcon().actualSize(QSize(size-border,size-border));
    painter->drawPixmap(-actualIconSize.width()/2, -actualIconSize.height()/2,
                        component->getIcon().pixmap(actualIconSize.width(),actualIconSize.height()));
    if(isSelected())
    {
        painter->setPen(QColor(200,200,200));
        painter->drawRect(boundingRect());
    }
}

PipelineComponent* AnalysisComponentSceneItem::getComponent()
{
    return component;
}
