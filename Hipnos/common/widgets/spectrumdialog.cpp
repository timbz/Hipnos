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

#include "spectrumdialog.h"
#include "common/filesysteminterface.h"

#include <QGridLayout>
#include <QLabel>
#include <QHeaderView>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>

LoadPresetDialog::LoadPresetDialog(Spectrum* s, QWidget *p)
    : QDialog(p)
{
    setWindowIcon(QIcon(":/icons/app-icon.png"));
    setMinimumWidth(300);
    spectrum = s;

    QGridLayout* l = new QGridLayout(this);
    setLayout(l);

    int row = 0;
    QLabel* title = new QLabel("<h2>Load preset</h2>");
    title->setStyleSheet("margin: 10px 0");
    title->setAlignment(Qt::AlignHCenter);
    l->addWidget(title, row++, 0, 1, 2);

    funtion = new QComboBox;
    l->addWidget(funtion, row++, 0, 1, 2);
    funtion->addItem("Gauss");
    funtion->addItem("Block");
    funtion->addItem("Sech");

    resolution = new QSpinBox;
    l->addWidget(resolution, row++, 0, 1, 2);
    resolution->setPrefix("Resolution: ");
    resolution->setMinimum(1);
    resolution->setMaximum(std::numeric_limits<int>::max());
    resolution->setValue(20);

    center = new QDoubleSpinBox;
    l->addWidget(center, row++, 0, 1, 2);
    center->setPrefix("Center: ");
    center->setSuffix(" 1/m");
    center->setDecimals(8);
    center->setMinimum(-std::numeric_limits<double>::max());
    center->setMaximum(std::numeric_limits<double>::max());
    center->setValue(1.0/(1030 * 0.000000001));

    width = new QDoubleSpinBox;
    l->addWidget(width, row++, 0, 1, 2);
    width->setDecimals(8);
    width->setPrefix("Width: ");
    width->setSuffix(" 1/m");
    width->setMinimum(-std::numeric_limits<double>::max());
    width->setMaximum(std::numeric_limits<double>::max());
    width->setValue( 1.0/(1030 * 0.000000001)/5);

    QPushButton* cancel = new QPushButton("Cancel");
    l->addWidget(cancel, row, 0, 1, 1);
    connect(cancel, SIGNAL(clicked()), this, SLOT(close()));

    QPushButton* ok = new QPushButton("Load");
    l->addWidget(ok, row++, 1 , 1, 1);
    connect(ok, SIGNAL(clicked()), this, SLOT(onOkClicked()));
    ok->setDefault(true);
}

void LoadPresetDialog::onOkClicked()
{
    if(funtion->currentText() == "Gauss")
        Spectrum::generateGaussianSpectrum(spectrum, resolution->value(), center->value(), width->value());
    else if(funtion->currentText() == "Block")
        Spectrum::generateBlockSpectrum(spectrum, resolution->value(), center->value(), width->value());
    else if(funtion->currentText() == "Sech")
        Spectrum::generateSechSpectrum(spectrum, resolution->value(), center->value(), width->value());
    close();
}

SpectrumDialog::SpectrumDialog(Spectrum s)
    : QDialog()
{
    setWindowIcon(QIcon(":/icons/app-icon.png"));
    spectrum = s;
    validator = new QDoubleValidator(this);
    QGridLayout* l = new QGridLayout(this);
    setLayout(l);
    QLabel* title = new QLabel("<h2>Spectrum editor</h2>");
    title->setStyleSheet("margin: 10px 0");
    title->setAlignment(Qt::AlignHCenter);
    l->addWidget(title, 0, 0,1, 2);

    table = new QTableWidget();
    l->addWidget(table, 1, 0, 1, 1);
    table->setShowGrid(true);
    table->setColumnCount(3);
    table->setMaximumWidth(250);
    table->setMinimumWidth(250);
    QStringList headers;
    headers << "Frequency" << "Intensity" << "";
    table->setHorizontalHeaderLabels(headers);
    table->horizontalHeader()->setResizeMode(0,QHeaderView::Stretch);
    table->horizontalHeader()->setResizeMode(1,QHeaderView::Stretch);
    table->horizontalHeader()->setResizeMode(1,QHeaderView::Fixed);
    table->setColumnWidth(2, 24);
    connect(table, SIGNAL(cellClicked(int,int)), this, SLOT(onTableCellClicked(int,int)));

    updateTable();

    plot = new QCustomPlot(this);
    plot->setStyleSheet("border: 1px solid black; min-width: 500px; min-height: 400px");
    plot->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    l->addWidget(plot, 1, 1, 1, 1);
    plot->addGraph();
    plot->graph(0)->setPen(QPen(Qt::blue));
    plot->graph(0)->setBrush(QBrush(QColor(0,0,255,150)));
    plot->addGraph();
    plot->graph(1)->setPen(QPen(Qt::yellow));
    plot->xAxis->setLabel("Frequency [1/m]");
    plot->yAxis->setLabel("Intensity []");

    updatePlot();

    QHBoxLayout* tableCtrlLayout = new QHBoxLayout;
    l->addLayout(tableCtrlLayout, 2, 0, 1, 1);
    QPushButton* loadCsv = new QPushButton("Load CSV");
    tableCtrlLayout->addWidget(loadCsv);
    connect(loadCsv, SIGNAL(clicked()), this, SLOT(onLoadCSVClicked()));
    QPushButton* loadPreset = new QPushButton("Load preset");
    tableCtrlLayout->addWidget(loadPreset);
    connect(loadPreset, SIGNAL(clicked()), this, SLOT(onLoadPresetClicked()));

    QPushButton* done = new QPushButton("Done");
    done->setDefault(true);
    connect(done, SIGNAL(clicked()), this, SLOT(close()));
    l->addWidget(done, 2, 1, 1, 1, Qt::AlignRight);
}

Spectrum SpectrumDialog::getSpectrum()
{
    return spectrum;
}

void SpectrumDialog::updateTable()
{
    disconnect(table, SIGNAL(cellChanged(int,int)), this, SLOT(onTableCellChanged(int,int)));
    table->clearContents();
    int row = 0;
    table->setRowCount(spectrum.size() + 1);
    foreach(Spectrum::SpectrumEntry e, spectrum.getEntries())
    {
        table->setItem(row, 0, new QTableWidgetItem(QString::number(e.Frequency)));
        table->setItem(row, 1, new QTableWidgetItem(QString::number(e.Intensity)));
        QTableWidgetItem* del = new QTableWidgetItem;
        del->setIcon(QIcon(":/icons/delete-black.png"));
        table->setItem(row, 2, del);
        row++;
    }
    connect(table, SIGNAL(cellChanged(int,int)), this, SLOT(onTableCellChanged(int,int)));
}

void SpectrumDialog::updatePlot()
{
    QVector<double> x;
    QVector<double> y;

    if(spectrum.size() > 0)
    {
        x << spectrum.getEntry(0).Frequency;
        y << spectrum.getEntry(0).Intensity;
        if(spectrum.size() > 1)
        {
            for (int i = 1; i < spectrum.size(); i++)
            {
                double intersection = spectrum.getEntry(i-1).Frequency + (spectrum.getEntry(i).Frequency - spectrum.getEntry(i-1).Frequency) / 2.0;
                x << intersection;
                y <<  spectrum.getEntry(i).Intensity;
                x << intersection;
                y <<  spectrum.getEntry(i-1).Intensity;
            }
            x << spectrum.getEntry(spectrum.size()-1).Frequency;
            y << spectrum.getEntry(spectrum.size()-1).Intensity;
        }
    }

    plot->graph(0)->setData(x, y);
    plot->graph(0)->rescaleAxes();

    QVector<double> x2(spectrum.size());
    QVector<double> y2(spectrum.size());
    for(int i = 0; i < spectrum.size(); i++)
    {
        x2[i] = spectrum.getEntry(i).Frequency;
        y2[i] = spectrum.getEntry(i).Intensity;
    }
    plot->graph(1)->setData(x2, y2);

    plot->yAxis->setRangeLower(0);
    plot->replot();
}

void SpectrumDialog::onTableCellChanged(int i, int j)
{
    int pos;
    QString newValue = table->item(i, j)->text();
    QValidator::State state = validator->validate(newValue, pos);

    if(state == QValidator::Acceptable)
    {
        if(i >= spectrum.size())
            spectrum.getEntries().resize(i+1);
        if(j == 0)
            spectrum.getEntry(i).Frequency = newValue.toDouble();
        else
            spectrum.getEntry(i).Intensity = newValue.toDouble();
    }
    spectrum.sort();
    updateTable();
    updatePlot();
}

void SpectrumDialog::onTableCellClicked(int i, int j)
{
     if(i < spectrum.size() && j == 2)
     {
         for(int k = i; k < spectrum.size()-1; k++)
         {
             spectrum.getEntry(k) = spectrum.getEntry(k+1);
         }
         spectrum.getEntries().resize(spectrum.size()-1);
         updateTable();
         updatePlot();
     }
}

void SpectrumDialog::onLoadCSVClicked()
 {
     QString filePath = QFileDialog::getOpenFileName(0, "Load CSV file", FileSystemInterface::getInstance().getHipnosHome().path(), "CSV Files (*.csv)");
     if(!filePath.isEmpty() && !filePath.isNull())
     {
         QFile file(filePath);
         if(file.open(QIODevice::ReadOnly | QIODevice::Text))
         {
             int row = 0;
             QList<QPair<double, double> > vals;
             while (!file.atEnd())
             {
                QString line = file.readLine();
                QStringList stringVals = line.split(",");
                if(stringVals.size() == 2)
                {
                    vals << QPair<double,double>(stringVals.at(0).toDouble(), stringVals.at(1).toDouble());
                }
                else
                {
                    QMessageBox::warning(this, "Error", "Invalid CSV (to many values on line " + QString::number(row).toLocal8Bit() + ")", QMessageBox::Close, QMessageBox::NoButton);
                    file.close();
                    return;
                }
                row++;
             }
             file.close();

             spectrum.getEntries().resize(vals.size());
             for(int i = 0; i < vals.size(); i++)
             {
                 spectrum.getEntry(i).Frequency = vals.at(i).first;
                 spectrum.getEntry(i).Intensity = vals.at(i).second;
             }
             updateTable();
             updatePlot();
         }
         else
         {
             QMessageBox::warning(this, "Error", "Could not open file " + filePath.toLocal8Bit(), QMessageBox::Close, QMessageBox::NoButton);
         }
     }
}

void SpectrumDialog::onLoadPresetClicked()
{
    LoadPresetDialog dia(&spectrum, this);
    dia.exec();
    updateTable();
    updatePlot();
}
