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

#include "chartsceneitem.h"
#include <QCursor>
#include <QGraphicsScene>

ChartSceneItem::ChartSceneItem(QGraphicsProxyWidget *la) :
    QGraphicsObject()
{
    pipeline = 0;
    pipelineLength = 0;
    yAxisSize = 130;
    xscale = 1;
    stepSize = 1;
    arrowSize.setWidth(8);
    arrowSize.setHeight(10);
    axisPadding = 10;
    loadingAnim = la;

    setCursor(Qt::PointingHandCursor);
    setAcceptedMouseButtons(Qt::LeftButton);
    setFlag(ItemIsSelectable);
}

ChartSceneItem::~ChartSceneItem()
{
}

void ChartSceneItem::setPipeline(Pipeline *p)
{
    pipeline = p;
}

void ChartSceneItem::updateChart(PipelineDataType dataType)
{
    wData.clear();
    pipelineComputationDataType = dataType;
    if(pipeline == 0)
        return;

    pipelineLength = pipeline->getLength();

    if(pipelineComputationDataType == DT_GAUSS)
    {
        wMax = 0;
        for(double step = 0; step <= pipelineLength; step += stepSize)
        {
            // W-Path
            PipelineSpectrumData* spectrumData = pipeline->propagation(DT_GAUSS, step);
            Spectrum spectrum = spectrumData->getSpectrum();
            double w = 0;
            for(int i = 0; i < spectrumData->getSize(); i++)
            {
                GaussPipelineData* data = spectrumData->getData<GaussPipelineData>(i);
                w += data->getW() * spectrum.getEntry(i).Intensity;
            }
            delete spectrumData;
            if(w > wMax)
                wMax = w;
            wData << w;
        }
    }    
    prepareGeometryChange();
    if(scene())
        scene()->update();
}

void ChartSceneItem::setXScale(double s)
{
    xscale = s;
    prepareGeometryChange();
    if(scene())
        scene()->update();
}

double ChartSceneItem::getXScale()
{
    return xscale;
}

void ChartSceneItem::setYAxisSize(double s)
{
    yAxisSize = s;
    prepareGeometryChange();
    if(scene())
        scene()->update();
}

double ChartSceneItem::getYAxisSize()
{
    return yAxisSize;
}

void ChartSceneItem::setStepSize(double s)
{
    stepSize = s;
    updateChart(pipelineComputationDataType);
}

double ChartSceneItem::getStepSize()
{
    return stepSize;
}

QRectF ChartSceneItem::boundingRect() const
{
    return QRectF(-axisPadding, -axisPadding - yAxisSize/2,
                  pipelineLength/xscale+2*axisPadding+arrowSize.height(),
                  (yAxisSize+2*axisPadding+arrowSize.height()));
}

void ChartSceneItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    if(wData.size() > 0)
    {
        wPolygon = QPolygonF();

        wYscale = wMax/yAxisSize*2;
        for(int i = 0; i < wData.size(); i++)
        {
            double step = stepSize*i/xscale;
            wPolygon << QPointF(step, -wData[i]/wYscale);
        }
        for(int i = wData.size()-1; i >= 0 ; i--)
        {
            double step = stepSize*i/xscale;
            wPolygon << QPointF(step, wData[i]/wYscale);
        }
        wPolygon << wPolygon.last();
        painter->setPen(QPen(QColor(255,200,0), 1, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        painter->setBrush(QBrush(QColor(255,200,0, 80)));
        painter->drawPolygon(wPolygon);
    }

    double scaledPipelineLength = pipelineLength/xscale;

    // draw axis
    painter->setPen(QPen(Qt::black, 1, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter->setBrush(QBrush(QColor(0,0,0)));
    painter->drawLine(-axisPadding,0, scaledPipelineLength+axisPadding,0);
    QPointF arrow2[3];
    arrow2[0] = QPointF(scaledPipelineLength+axisPadding+arrowSize.height(),0);
    arrow2[1] = QPointF(scaledPipelineLength+axisPadding,-arrowSize.width()/2);
    arrow2[2] = QPointF(scaledPipelineLength+axisPadding,arrowSize.width()/2);
    painter->drawPolygon(arrow2, 3);

//    painter->drawLine(0,axisPadding,0,-yAxisSize-axisPadding);
//    QPointF arrow1[3];
//    arrow1[0] = QPointF(0,-yAxisSize-axisPadding-arrowSize.height());
//    arrow1[1] = QPointF(-arrowSize.width()/2,-yAxisSize-axisPadding);
//    arrow1[2] = QPointF(arrowSize.width()/2,-yAxisSize-axisPadding);
//    painter->drawPolygon(arrow1, 3);

//    painter->setBrush(Qt::NoBrush);
//    painter->drawRect(boundingRect());
}
