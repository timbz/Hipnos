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

#include <QSpacerItem>
#include <QFileDialog>
#include <QDebug>
#include <QMutexLocker>
#include <QMovie>

#include "analysiscontroller.h"
#include "common/filesysteminterface.h"
#include "analysiscomponentsceneitem.h"
#include "chartsceneitem.h"
#include "exporttocsvdialog.h"

AnalysisController::AnalysisController(Ui::MainWindow *ui) :
    QObject()
{
    pipeline = 0;
    vtkWidget = ui->qvtkWidget;
    currentThread = 0;
    nextThread = 0;
    sliderPosLabel = ui->sliderSceneXPosLabel;

    connect(ui->exportToCsv, SIGNAL(clicked()), this, SLOT(exportToCsv()));

    loadingAnimation = new QLabel();
    QMovie *movie = new QMovie(":/icons/animations/loading-black.gif");
    movie->setParent(loadingAnimation);
    loadingAnimation->setMovie(movie);
    movie->setScaledSize(QSize(50,50));
    movie->start();
    loadingAnimation->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
    loadingAnimation->setStyleSheet("background-color: black; padding: 10px; border: 1px solid white;");
    vtkWidget->setStyleSheet("background-color: black;");

    QHBoxLayout* l = new QHBoxLayout(vtkWidget);
    vtkWidget->setLayout(l);
    l->addWidget(loadingAnimation);
    l->setAlignment(loadingAnimation, Qt::AlignBottom | Qt::AlignLeft);
    loadingAnimation->hide();

    // Init VTK
    vtkPipeline = new VtkPipeline(vtkWidget->GetRenderWindow());

    // Init slider
    sliderScene = new AnalysisSliderScene(ui->analysisSlider);
    connect(sliderScene, SIGNAL(sliderMoved(double)), this, SLOT(onSliderChanged(double)));
    connect(sliderScene, SIGNAL(selectionChanged()), this, SLOT(onSliderSceneSelectionChanged()));

    // Init Component Property Frame
    if(!ui->analysePropertyFrame->layout())
    {
        QVBoxLayout* propertyFrameLayout = new QVBoxLayout(ui->analysePropertyFrame);
        propertyFrameLayout->setSpacing(0);
        propertyFrameLayout->setContentsMargins(0, 0, 0, 0);
    }
    componentPropertyWidget = new ComponentPropertyWidget();
    ui->analysePropertyFrame->layout()->addWidget(componentPropertyWidget);
    componentPropertyWidget->setXScale(sliderScene->getXScale());
    connect(componentPropertyWidget, SIGNAL(propertyChanged(PipelineComponent::Property)), this, SLOT(onPropertyChanged(PipelineComponent::Property)));
    connect(componentPropertyWidget, SIGNAL(xScaleChanged(double)), sliderScene, SLOT(setXScale(double)));

    // Init VTK view Property Frame
    if(!ui->analysisViewPropertyFrame->layout())
    {
        QVBoxLayout* layout = new QVBoxLayout(ui->analysisViewPropertyFrame);
        layout->setSpacing(0);
        layout->setContentsMargins(0, 0, 0, 0);
    }
    vtkViewPropertyWidget = new VtkViewPropertyWidget();
    ui->analysisViewPropertyFrame->layout()->addWidget(vtkViewPropertyWidget);
    vtkPipeline->setColorTransferFunction(vtkViewPropertyWidget->getGradient(),
                             vtkViewPropertyWidget->getGradientMinValue(),
                             vtkViewPropertyWidget->getGradientMaxValue());
    vtkPipeline->setWarpingMapping(vtkViewPropertyWidget->getWarpingMapping());
    vtkPipeline->setColorMapping(vtkViewPropertyWidget->getColorMapping());
    vtkPipeline->setWarpingScaling(vtkViewPropertyWidget->getWarpingScaling());
    vtkPipeline->setFrequencyPercentageInterval(vtkViewPropertyWidget->getFrequencyMinPercentage(), vtkViewPropertyWidget->getFrequencyMaxPercentage());
    connect(vtkViewPropertyWidget, SIGNAL(gradientChanged(QGradient,float,float)), vtkPipeline, SLOT(setColorTransferFunction(QGradient,float,float)));
    connect(vtkViewPropertyWidget, SIGNAL(warpingMappingChanged(VtkViewPropertyWidget::MappingFunction)), vtkPipeline, SLOT(setWarpingMapping(VtkViewPropertyWidget::MappingFunction)));
    connect(vtkViewPropertyWidget, SIGNAL(gradientMappingChanged(VtkViewPropertyWidget::MappingFunction)), vtkPipeline, SLOT(setColorMapping(VtkViewPropertyWidget::MappingFunction)));
    connect(vtkViewPropertyWidget, SIGNAL(warpingScalingChanged(double)), vtkPipeline, SLOT(setWarpingScaling(double)));
    connect(vtkViewPropertyWidget, SIGNAL(applyProperties()), vtkPipeline, SLOT(render()));
    connect(vtkPipeline, SIGNAL(dataBoundsChanged(double,double)), vtkViewPropertyWidget, SLOT(setGradientMinMax(double,double)));
    connect(vtkPipeline, SIGNAL(spectrumChanged(Spectrum)), vtkViewPropertyWidget, SLOT(setSpectrum(Spectrum)), Qt::BlockingQueuedConnection); // signal from different thread
    connect(vtkViewPropertyWidget, SIGNAL(frequencyPercentageRangeChanged(int,int)), this, SLOT(onFrequencyPercentageRangeChanged(int,int)));
}

AnalysisController::~AnalysisController()
{
    if(pipeline)
        delete pipeline;
    if(nextThread)
    {
        delete nextThread;
    }
    if(currentThread)
    {
        currentThread->wait();
        delete currentThread;
    }
    delete vtkPipeline;
    delete sliderScene;
}

void AnalysisController::setPipeline(Pipeline *p)
{
    pipeline = p;
    sliderScene->drawPipeline(p);
    vtkPipeline->setResetCamera();
    createGausPropagationThread(0);
}

AnalysisController::PropagationThread::PropagationThread(AnalysisController *c, double z) :
    QThread()
{
    zValue = z;
    controller = c;
}

void AnalysisController::PropagationThread::run()
{
    controller->computePropagationAt(zValue);
}

void AnalysisController::computePropagationAt(double z)
{
    if(!pipeline)
        return;
    QTime timer;
    timer.start();
    PipelineSpectrumData* spectrum = pipeline->propagation(pipelineComputationDataType, z);
    qDebug() << "Simulation:" << timer.elapsed() << "ms";

    timer.restart();
    vtkPipeline->update(spectrum);
    qDebug() << "VTK update:" << timer.elapsed() << "ms";

    delete spectrum;
}

void AnalysisController::createGausPropagationThread(double z)
{
    sliderPosLabel->setText("Slider position: " + QString::number(z) + " m");
    PropagationThread* t = new PropagationThread(this, z);
    connect(t, SIGNAL(finished()), this, SLOT(onThreadFinished()));
    if(currentThread == 0)
    {
        currentThread = t;
        // Disable widget interaction to avoid concurrent pipeline updates
        vtkWidget->GetInteractor()->Disable();
        // Show loading img
        loadingAnimation->show();
        // Start the thread
        currentThread->start();
    }
    else
    {
        if(nextThread)
            delete nextThread;
        nextThread = t;
    }
}

void AnalysisController::onThreadFinished()
{
    delete currentThread;
    if(nextThread)
    {
        currentThread = nextThread;
        nextThread = 0;
        currentThread->start();
    }
    else
    {
        sliderScene->updatePipeline(pipelineComputationDataType);
        currentThread = 0;
        // render

        QTime timer;
        timer.start();
        vtkPipeline->render();
        qDebug() << "VTK render:" << timer.elapsed() << " ms";
        // Enable interaction if we are done with all calculations
        vtkWidget->GetInteractor()->Enable();
        // Hide loading img
        loadingAnimation->hide();
    }
}

void AnalysisController::onPropertyChanged(PipelineComponent::Property p)
{
    sliderScene->markForUpdate();
    createGausPropagationThread(sliderScene->getSliderValue());
}

void AnalysisController::onFrequencyPercentageRangeChanged(int min, int max)
{
    vtkPipeline->setFrequencyPercentageInterval(min, max);
    createGausPropagationThread(sliderScene->getSliderValue());
}

void AnalysisController::onSliderChanged(double z)
{
    createGausPropagationThread(z);
}

void AnalysisController::onSliderSceneSelectionChanged()
{
    if(sliderScene->selectedItems().size() > 0)
    {
        AnalysisComponentSceneItem* item = dynamic_cast<AnalysisComponentSceneItem*>(sliderScene->selectedItems().first());
        if(item)
        {
            componentPropertyWidget->setComponent(item->getComponent());
            return;
        }
        ChartSceneItem* chart = dynamic_cast<ChartSceneItem*>(sliderScene->selectedItems().first());
        if(chart)
        {
            componentPropertyWidget->setChart(chart);
        }
    }
    else
    {
        //unselect
        componentPropertyWidget->reset();
    }
}

void AnalysisController::exportSliderSceneAsJpeg()
{
    QString filePath = QFileDialog::getSaveFileName(0, "Save...", FileSystemInterface::getInstance().getHipnosHome().path(), "Jpeg Files (*.jpg)");
    if(!filePath.isEmpty() && !filePath.isNull())
    {
        if(!filePath.endsWith(".jpg"))
            filePath += ".jpg";
        sliderScene->toImage().save(filePath, "JPG");
    }
}

void AnalysisController::exportToCsv()
{
    ExportToCsvDialog dia(pipeline, pipelineComputationDataType, sliderScene->getSliderValue());
    dia.exec();
}

void AnalysisController::exportVtkAsJpeg()
{
    QString filePath = QFileDialog::getSaveFileName(0, "Save...", FileSystemInterface::getInstance().getHipnosHome().path(), "Jpeg Files (*.jpg)");
    vtkPipeline->exportAsJpeg(filePath);
}

void AnalysisController::sliderSceneZoomIn()
{
    sliderScene->zoomIn();
}

void AnalysisController::sliderSceneZoomOut()
{
    sliderScene->zoomOut();
}

void AnalysisController::setPipelineComputationDataType(PipelineDataType dataType)
{
    if(pipelineComputationDataType == dataType)
        return;
    pipelineComputationDataType = dataType;
    sliderScene->markForUpdate();
    createGausPropagationThread(sliderScene->getSliderValue());
}
