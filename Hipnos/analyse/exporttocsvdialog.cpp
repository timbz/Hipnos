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

#include "exporttocsvdialog.h"
#include "common/filesysteminterface.h"
#include <QGridLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QSpacerItem>
#include <QLabel>
#include <QPushButton>
#include <QMessageBox>
#include <QButtonGroup>

ExportToCsvDialog::ExportToCsvDialog(Pipeline *p, PipelineDataType dt, double z, QWidget *parent) :
    QDialog(parent)
{
    setWindowIcon(QIcon(":/icons/app-icon.png"));

    pipeline = p;
    dataType = dt;
    zValue = z;


    QGridLayout* layout = new QGridLayout;
    layout->setSpacing(20);
    setLayout(layout);

    QLabel* titleLabel = new QLabel("<h2>Export to CSV</h2>");
    layout->addWidget(titleLabel, 0, 0, 1, 3);
    titleLabel->setAlignment(Qt::AlignHCenter);
    QSpacerItem* titleSpacer = new QSpacerItem(7, 20);
    layout->addItem(titleSpacer, 1, 0);

    QLabel* fileNameLabel = new QLabel("<b>File name:</b>");
    layout->addWidget(fileNameLabel, 2, 0);
    fileNameEdit = new QLineEdit();
    layout->addWidget(fileNameEdit, 2, 1);
    QPushButton* chooseFileButton = new QPushButton("Browse");
    layout->addWidget(chooseFileButton, 2, 2);
    connect(chooseFileButton, SIGNAL(clicked()), this, SLOT(onChooseFile()));

    QLabel* dataSelectLabel = new QLabel("<b>Data:</b>");
    layout->addWidget(dataSelectLabel, 3, 0);
    QButtonGroup* dataSelectGroup = new QButtonGroup();
    QHBoxLayout* dataSelectLayout = new QHBoxLayout;
    layout->addLayout(dataSelectLayout, 3, 1, 1, 2);
    realButton = new QRadioButton("Real{E}");
    realButton->setChecked(true);
    dataSelectGroup->addButton(realButton);
    dataSelectLayout->addWidget(realButton);
    imagButton = new QRadioButton("Imag{E}");
    dataSelectGroup->addButton(imagButton);
    dataSelectLayout->addWidget(imagButton);
    absButton = new QRadioButton(trUtf8("Abs{E}²"));
    dataSelectGroup->addButton(absButton);
    dataSelectLayout->addWidget(absButton);
    argButton = new QRadioButton("Arg{E}");
    dataSelectGroup->addButton(argButton);
    dataSelectLayout->addWidget(argButton);

    QLabel* dataPrecisionLabel = new QLabel("<b>Precision:</b>");
    layout->addWidget(dataPrecisionLabel, 4, 0);
    dataPrecisionSpinBox = new QSpinBox;
    layout->addWidget(dataPrecisionSpinBox, 4, 1);
    dataPrecisionSpinBox->setValue(5);
    dataPrecisionSpinBox->setSuffix(" decimal digits");

    QLabel* dellimiterLabel = new QLabel("<b>Delimiter:</b>");
    layout->addWidget(dellimiterLabel, 5, 0);
    delimiterEdit = new QLineEdit(",");
    layout->addWidget(delimiterEdit, 5, 1);

    QLabel* dataFormatLabel = new QLabel("<b>Format:</b>");
    layout->addWidget(dataFormatLabel, 6, 0);
    QButtonGroup* dataFormatGroup = new QButtonGroup();
    QHBoxLayout* dataFormatLayout = new QHBoxLayout;
    layout->addLayout(dataFormatLayout, 6, 1);
    normalButton = new QRadioButton("Normal (5.034)");
    normalButton->setChecked(true);
    dataFormatGroup->addButton(normalButton);
    dataFormatLayout->addWidget(normalButton);
    scientificButton = new QRadioButton("Scientific (7.0E-5)");
    dataFormatGroup->addButton(scientificButton);
    dataFormatLayout->addWidget(scientificButton);

    QSpacerItem* spacer = new QSpacerItem(1, 60);
    layout->addItem(spacer, 7, 0);

    QHBoxLayout* defaulButtonsLayout = new QHBoxLayout;
    layout->addLayout(defaulButtonsLayout, 8, 0, 1, 3);
    QSpacerItem* defaulButtonsLayoutSpacer = new QSpacerItem(1 ,1 , QSizePolicy::Expanding);
    defaulButtonsLayout->addItem(defaulButtonsLayoutSpacer);
    QPushButton* cancelButton = new QPushButton("Cancel");
    defaulButtonsLayout->addWidget(cancelButton);
    connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
    QPushButton *exportButton = new QPushButton("Export");
    defaulButtonsLayout->addWidget(exportButton);
    connect(exportButton, SIGNAL(clicked()), this, SLOT(exportToCsv()));
    exportButton->setDefault(true);
}

void ExportToCsvDialog::onChooseFile()
{
    QString filePath = QFileDialog::getSaveFileName(0, "Save...", FileSystemInterface::getInstance().getHipnosHome().path(), "CSV Files (*.csv)");
    if(!filePath.isEmpty() && !filePath.isNull())
    {
        if(!filePath.endsWith(".csv"))
            filePath += ".csv";
        fileNameEdit->setText(filePath);
    }
}

double real(const std::complex<double>& a)
{
    return a.real();
}
double imag(const std::complex<double>& a)
{
    return a.imag();
}
double abs(const std::complex<double>& a)
{
    return std::abs(a);
}
double arg(const std::complex<double>& a)
{
    return std::arg(a);
}

void ExportToCsvDialog::exportToCsv()
{
    if(pipeline == 0)
    {
        QMessageBox::warning(this, "Error", "Invalid pipeline", QMessageBox::Close, QMessageBox::NoButton);
        close();
        return;
    }

    PipelineSpectrumData* spectrum = pipeline->propagation(dataType, zValue);
    if(spectrum == 0)
    {
        QMessageBox::warning(this, "Error", "Invalid pipeline data", QMessageBox::Close, QMessageBox::NoButton);
        close();
        return;
    }

    QFile file(fileNameEdit->text());
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        QMessageBox::warning(this, "Error", "Could not open file " + fileNameEdit->text(), QMessageBox::Close, QMessageBox::NoButton);
        return;
    }

    double (*func)(const std::complex<double>&) = 0;
    if(realButton->isChecked())
        func = &real;
    else if(imagButton->isChecked())
        func = &imag;
    else if(absButton->isChecked())
        func = &abs;
    else if(argButton->isChecked())
        func = &arg;
    if(func == 0)
    {
        QMessageBox::warning(this, "Error", "No data selected", QMessageBox::Close, QMessageBox::NoButton);
        return;
    }

    Matrix* result = spectrum->getComplexAmplitude();
    delete spectrum;

    QString delimiter = delimiterEdit->text();
    int precision = dataPrecisionSpinBox->value();
    char format = 'f';
    if(scientificButton->isChecked())
        format = 'E';

    QTextStream out(&file);
    for( int i = 0; i < result->getRows(); i++)
    {
        for(int j = 0; j < result->getCols(); j++)
        {
            out  << QString::number(func(result->get(i, j)), format, precision);
            if(j < result->getCols()-1);
                out << delimiter;
        }
        if(i < result->getRows()-1)
            out << "\n";
    }

    file.close();
    close();

}
