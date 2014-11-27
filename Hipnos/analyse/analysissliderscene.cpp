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

#include "analysissliderscene.h"
#include "slidersceneitem.h"

#include <QDebug>
#include <QLabel>
#include <QMovie>

AnalysisSliderScene::AnalysisSliderScene(QGraphicsView* sv, QObject *parent) :
    QGraphicsScene(parent)
{
    sv->setScene(this);
    view = sv;
    dragging = false;
    clear();
    xscale = 1;
    componentIconSize = 60;
    pipeline = 0;
    updateFlag = false;
}

void AnalysisSliderScene::wheelEvent(QWheelEvent *event)
{
    scaleView(pow((double)2, -event->delta() / 240.0));
}

void AnalysisSliderScene::zoomIn()
{
    scaleView(qreal(1.2));
}

 void AnalysisSliderScene::zoomOut()
{
    scaleView(1 / qreal(1.2));
}

void AnalysisSliderScene::scaleView(qreal scaleFactor)
{
    qreal factor = view->transform().scale(scaleFactor, scaleFactor).mapRect(QRectF(0, 0, 1, 1)).width();
    if (factor < 0.07 || factor > 100)
        return;
    view->scale(scaleFactor, scaleFactor);
}

void AnalysisSliderScene::clear()
{
    QGraphicsScene::clear();
    componentItems.clear();
    sliderItem = new SliderSceneItem(60);
    sliderItem->setPos(0,0);
    sliderItem->setZValue(100);
    addItem(sliderItem);

    QLabel* loadingAnim = new QLabel();
    QMovie *movie = new QMovie(":/icons/animations/loading.gif");
    movie->setParent(this);
    loadingAnim->setMovie(movie);
    movie->start();
    loadingItem = addWidget(loadingAnim);
    loadingItem->setPos(-60,-8);
    loadingItem->hide();
}

void AnalysisSliderScene::setXScale(double s)
{
    xscale = s;
    chartItem->setXScale(s);
    if(pipeline)
    {
        scaledPipelineLength = pipeline->getLength()/xscale;
        updateComponentPos();
    }
}

qreal AnalysisSliderScene::getXScale()
{
    return xscale;
}

void AnalysisSliderScene::drawPipeline(Pipeline *p)
{
    pipeline = p;
    clear();
    scaledPipelineLength = p->getLength()/xscale;

    chartItem = new ChartSceneItem(loadingItem);
    chartItem->setXScale(xscale);
    chartItem->setPos(0, 0);
    chartItem->setPipeline(p);
    addItem(chartItem);
    markForUpdate();

    PipelineComponent* c = pipeline->getFirstComponent();
    while(c != 0)
    {
        AnalysisComponentSceneItem* item = new AnalysisComponentSceneItem(c, componentIconSize);
        addItem(item);
        componentItems << item;
        if(c->getNumberOfOutputConnections() > 0)
        {
            c = c->getOutputConnection()->getOutput();
        }
        else
        {
            c = 0;
        }
    }
    updateComponentPos();

}

void AnalysisSliderScene::markForUpdate()
{
    updateFlag = true;
}

void AnalysisSliderScene::updatePipeline(PipelineDataType pipelineComputationDataType)
{
    if(pipeline == 0 || !updateFlag)
        return;
    double newLength = pipeline->getLength()/xscale;
    if(newLength != scaledPipelineLength)
    {
        scaledPipelineLength = newLength;
        updateComponentPos();
    }
    // redraw chart
    chartItem->updateChart(pipelineComputationDataType);
    updateFlag = false;
}

void AnalysisSliderScene::updateComponentPos()
{
    double xoffset = 0;
    double yoffset = 0;
    double i = 0;
    PipelineComponent* c = pipeline->getFirstComponent();
    while(c != 0)
    {
        if(c->getLength() == 0)
        {
            componentItems[i]->setPos(xoffset, yoffset);
            yoffset += componentIconSize;
        }
        else
        {
            componentItems[i]->setPos(xoffset + c->getLength()/xscale/2, 0);
            xoffset += c->getLength()/xscale;
            yoffset = 0;
        }

        // propagate downstream
        if(c->getNumberOfOutputConnections() > 0)
        {
            c = c->getOutputConnection(0)->getOutput();
        }
        else
        {
            c = 0;
        }
        i++;
    }
}

void AnalysisSliderScene::mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    if(itemAt(mouseEvent->scenePos(), QTransform()) == sliderItem)
    {
        dragging = true;
        sliderItem->draggingStart();
        dragginOffset = sliderItem->pos() - mouseEvent->scenePos();
    }
    else
    {
        // disable multiple selection
        clearSelection();
        QGraphicsScene::mousePressEvent(mouseEvent);
    }
}

void AnalysisSliderScene::mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    if (dragging) {
        QPointF tmp = dragginOffset + mouseEvent->scenePos();
        tmp.setY(0);
        if(tmp.x() < 0)
        {
            tmp.setX(0);
        }
        if(tmp.x() > scaledPipelineLength)
        {
            tmp.setX(scaledPipelineLength);
        }
        if(sliderItem->pos() != tmp)
        {
            sliderItem->setPos(tmp);
            emit sliderMoved(getSliderValue());
        }
    } else
        QGraphicsScene::mouseMoveEvent(mouseEvent);
}

qreal AnalysisSliderScene::getSliderValue()
{
    return sliderItem->pos().x()*xscale;
}

void AnalysisSliderScene::mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    if (dragging) {
        dragging = false;
        sliderItem->draggingEnd();
        //emit sliderMoved(getSliderValue());
    } else
        QGraphicsScene::mouseReleaseEvent(mouseEvent);
}

QImage AnalysisSliderScene::toImage()
{
    clearSelection();
    sliderItem->hide();

    QImage image(width(), height(), QImage::Format_ARGB32);
    image.fill(Qt::white);
    QPainter p(&image);
    p.setBackgroundMode(Qt::OpaqueMode);
    p.setBackground(Qt::white);
    p.setRenderHint(QPainter::Antialiasing);
    render(&p);
    p.end();

    sliderItem->show();

    return image;
}
