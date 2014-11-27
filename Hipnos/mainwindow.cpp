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

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "common/statusmessage.h"
#include <QDebug>
#include <QMessageBox>

/**
 * @brief
 *
 * @param parent
 */
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowIcon(QIcon(":/icons/app-icon.png"));

    StatusMessage::getInstance().setStatusBar(ui->statusBar);

    designer = new DesignerController(ui);
    connect(ui->designerZoomOut, SIGNAL(clicked()), designer, SLOT(zoomOut()));
    connect(ui->designerZoomIn, SIGNAL(clicked()), designer, SLOT(zoomIn()));
    connect(ui->designerDelete, SIGNAL(clicked()), designer, SLOT(deleteSelectedItems()));
    connect(ui->designerGroup, SIGNAL(clicked()), designer, SLOT(groupSelectedItems()));
    connect(ui->designerUngroup, SIGNAL(clicked()), designer, SLOT(ungroupSelectedItems()));
    connect(ui->customComponentListDelete, SIGNAL(clicked()), designer, SLOT(deleteCustomComponent()));

    analyser = new AnalysisController(ui);
    connect(ui->vtkExportToImage, SIGNAL(clicked()), analyser, SLOT(exportVtkAsJpeg()));
    connect(ui->sliderSceneExportToImage, SIGNAL(clicked()), analyser, SLOT(exportSliderSceneAsJpeg()));
    connect(ui->sliderSceneZoomIn, SIGNAL(clicked()), analyser, SLOT(sliderSceneZoomIn()));
    connect(ui->sliderSceneZoomOut, SIGNAL(clicked()), analyser, SLOT(sliderSceneZoomOut()));

    // select designer as default view
    ui->designButton->setChecked(true);
    ui->designFrame->show();
    ui->analyseButton->setChecked(false);
    ui->analyseFrame->hide();
    ui->viewButton->setChecked(false);
    ui->viewFrame->hide();
    // hide the View Button for now1
    ui->viewButton->hide();

    ui->dataTypeSwitchFrame->setMaximumHeight(0);
    ui->dataTypeSwitchFrame->setMinimumHeight(0);
    ui->gaussDataType->setChecked(true);
    analyser->setPipelineComputationDataType(DT_GAUSS);
    QPropertyAnimation *anim1 = new QPropertyAnimation(ui->dataTypeSwitchFrame, "maximumHeight");
    anim1->setDuration(400);
    anim1->setStartValue(0);
    anim1->setEndValue(120);
    QPropertyAnimation *anim2 = new QPropertyAnimation(ui->dataTypeSwitchFrame, "minimumHeight");
    anim2->setDuration(400);
    anim2->setStartValue(0);
    anim2->setEndValue(120);
    dataTypeSwitchFrameAnimation = new QParallelAnimationGroup;
    dataTypeSwitchFrameAnimation->addAnimation(anim1);
    dataTypeSwitchFrameAnimation->addAnimation(anim2);
}

MainWindow::~MainWindow()
{
    delete designer;
    delete analyser;
    delete ui;
    delete dataTypeSwitchFrameAnimation;
    Math::destroy();
}

/**
 * @brief
 *
 * @return bool
 */
bool MainWindow::validateDesignedPipeline()
{
    if(ui->designButton->isChecked())
    {
        bool pipelineModified = designer->getPipelineModified();
        Pipeline* p = designer->getPipeline();
        if(!p)
        {
            ui->analyseButton->setChecked(false);
            ui->viewButton->setChecked(false);
            StatusMessage::show("Pipeline is not valid");
            return false;
        }
        if(pipelineModified)
            analyser->setPipeline(p);
    }
    return true;
}

void MainWindow::showDataTypeSwitchFrame()
{
    if(ui->dataTypeSwitchFrame->minimumHeight() < 120)
    {
        dataTypeSwitchFrameAnimation->setDirection(QAbstractAnimation::Forward);
        dataTypeSwitchFrameAnimation->start();
    }
}

void MainWindow::hideDataTypeSwitchFrame()
{
    if(ui->dataTypeSwitchFrame->minimumHeight() > 0)
    {
        dataTypeSwitchFrameAnimation->setDirection(QAbstractAnimation::Backward);
        dataTypeSwitchFrameAnimation->start();
    }
}

void MainWindow::on_designButton_clicked()
{
    ui->designButton->setChecked(true);
    if(ui->designFrame->isHidden())
    {
        ui->analyseButton->setChecked(false);
        ui->analyseFrame->hide();
        ui->viewButton->setChecked(false);
        ui->viewFrame->hide();
        ui->designFrame->show();
        hideDataTypeSwitchFrame();
    }
}

void MainWindow::on_analyseButton_clicked()
{
    if(!validateDesignedPipeline()) return;
    ui->analyseButton->setChecked(true);
    if(ui->analyseFrame->isHidden())
    {
        ui->viewButton->setChecked(false);
        ui->viewFrame->hide();
        ui->designButton->setChecked(false);
        ui->designFrame->hide();
        ui->analyseFrame->show();
        showDataTypeSwitchFrame();
    }
}

void MainWindow::on_viewButton_clicked()
{
    if(!validateDesignedPipeline()) return;
    ui->viewButton->setChecked(true);
    if(ui->viewFrame->isHidden())
    {
        ui->designButton->setChecked(false);
        ui->designFrame->hide();
        ui->analyseButton->setChecked(false);
        ui->analyseFrame->hide();
        ui->viewFrame->show();
        hideDataTypeSwitchFrame();
    }
}

void MainWindow::on_actionQuit_triggered()
{
    QApplication::exit(0);
}

void MainWindow::on_actionSave_triggered()
{
    designer->save();
}

void MainWindow::on_actionOpen_triggered()
{
    ui->designButton->setChecked(true);
    ui->designFrame->show();
    ui->analyseButton->setChecked(false);
    ui->analyseFrame->hide();
    ui->viewButton->setChecked(false);
    ui->viewFrame->hide();
    designer->load();
}

void MainWindow::on_actionAbout_Hipnos_triggered()
{
    QString content = "";
    content += "<div>";
    content +=  "<b>HI</b> gh<br/>";
    content +=  "<b>P</b> erformance<br/>";
    content +=  "<b>N</b> on linear<br/>";
    content +=  "<b>O</b> ptics<br/>";
    content +=  "<b>S</b> imulation<br/>";
    content += "</div>";
    content += "<div>";
    content +=  "<i><font size=\"8px\">Icons from <a href=\"http://gentleface.com\">gentleface.com</a> - Creative Commons Attribution-Noncommercial Works 3.0 Unported license</font></i>";
    content += "</div>";

    QMessageBox::about(this, "About Hipnos", content);
}

void MainWindow::on_gaussDataType_clicked()
{
    ui->gaussDataType->setChecked(true);
    ui->fourierDataType->setChecked(false);
    analyser->setPipelineComputationDataType(DT_GAUSS);
}

void MainWindow::on_fourierDataType_clicked()
{
    ui->gaussDataType->setChecked(false);
    ui->fourierDataType->setChecked(true);
    analyser->setPipelineComputationDataType(DT_FOURIER);
}
