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

#ifndef PROPERTYCHANGEDHANDLER_H
#define PROPERTYCHANGEDHANDLER_H

#include "common/pipelinecomponent.h"
#include "common/filesysteminterface.h"
#include "common/widgets/spectrumdialog.h"
#include <QDebug>
#include <QLineEdit>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>


/**
 * @brief Instances of this class act as a proxy between an imput widget (like a QLineEdit) and a PipelineComponent::Property.
 *        They recieve the user imput from the widgets and apply them to the referenced property
 */
class PropertyChangedHandler : public QObject
{
    Q_OBJECT

public:

    /**
     * @brief Connects a PipelineComponent::Property with a QLineEdit
     *
     * @param p The property of type PipelineComponent::Property::PT_STRING
     * @param le A reference to the QLineEdit used to edit this property
     */
    PropertyChangedHandler(PipelineComponent::Property p, QLineEdit* le) :
        QObject(),
        prop(p),
        tmpFilePahtLineEdit(0)
    {
        connect(le, SIGNAL(textEdited(QString)), this, SLOT(onTextEdited(QString)));
    }

    /**
     * @brief Connects a PipelineComponent::Property with a QSpinBox
     *
     * @param p The property of type PipelineComponent::Property::PT_INT
     * @param sb A reference to the QSpinBox used to edit this property
     */
    PropertyChangedHandler(PipelineComponent::Property p, QSpinBox* sb) :
        QObject(),
        prop(p),
        tmpFilePahtLineEdit(0)
    {
        connect(sb, SIGNAL(valueChanged(int)), this, SLOT(onSpinBoxChanged(int)));
    }

    /**
     * @brief Connects a PipelineComponent::Property with a QDoubleSpinBox
     *
     * @param p The property of type PipelineComponent::Property::PT_DOUBLE
     * @param sb A reference to the QDoubleSpinBox used to edit this property
     */
    PropertyChangedHandler(PipelineComponent::Property p, QDoubleSpinBox* sb) :
        QObject(),
        prop(p),
        tmpFilePahtLineEdit(0)
    {
        connect(sb, SIGNAL(valueChanged(double)), this, SLOT(onSpinBoxChanged(double)));
    }

    /**
     * @brief Connects a PipelineComponent::Property with a QComboBox
     *
     * @param p The property of type PipelineComponent::Property::PT_LIST
     * @param cb A reference to the QComboBox used to edit this property
     */
    PropertyChangedHandler(PipelineComponent::Property p, QComboBox* cb) :
        QObject(),
        prop(p),
        tmpFilePahtLineEdit(0)
    {
        connect(cb, SIGNAL(activated(QString)), this, SLOT(onComboBoxChanged(QString)));
    }

    /**
     * @brief Connects a PipelineComponent::Property with a QFileDialog
     *
     * @param p The property of type PipelineComponent::Property::PT_FILE
     * @param lineEdit A reference to the QLineEdit used to display the current file name
     * @param browse A reference to the QPushButton used to open the QFileDialog
     */
    PropertyChangedHandler(PipelineComponent::Property p, QLineEdit* lineEdit, QPushButton* browse) :
        QObject(),
        prop(p)
    {
        connect(lineEdit, SIGNAL(textChanged(QString)), this, SLOT(onFilePathEdited(QString)));
        tmpFilePahtLineEdit = lineEdit;
        connect(browse, SIGNAL(clicked()), this, SLOT(openFileDialog()));
    }

    /**
     * @brief Connects a PipelineComponent::Property with a SpectrumDialog
     *
     * @param p The property of type PipelineComponent::Property::PT_SPECTRUM
     * @param b A reference to the QPushButton used to open the SpectrumDialog
     */
    PropertyChangedHandler(PipelineComponent::Property p, QPushButton* b) :
        QObject(),
        prop(p)
    {
        connect(b, SIGNAL(clicked()), this, SLOT(onSpectrumEdit()));
    }

signals:
    /**
     * @brief Signal emitted when the user changes the referenced PipelineComponent::Property
     *
     * @param p The changed PipelineComponent::Property
     */
    void propertyChanged(PipelineComponent::Property p);

private slots:
    /**
     * @brief Internal slot used to catch the QLineEdit::textEdited() signal
     *
     * @param s The new value
     */
    void onTextEdited(QString s)
    {
        prop.setValue(s);
        prop.getComponent()->enqueueProperty(prop);
        prop.getComponent()->setChanged();
        emit propertyChanged(prop);
    }

    /**
     * @brief Internal slot used to catch the QLineEdit::textChanged() signal
     *
     * @param s The new value
     */
    void onFilePathEdited(QString s)
    {
        QFile f(s);
        if(f.exists())
        {
            prop.setFile(f);
            prop.getComponent()->enqueueProperty(prop);
            prop.getComponent()->setChanged();
            emit propertyChanged(prop);
        }
        else
            QMessageBox::warning(0, "Error", "Could not open file " + s, QMessageBox::Close, QMessageBox::NoButton);
    }

    /**
     * @brief Internal slot used to open a QFileDialog
     *
     */
    void openFileDialog()
    {
        if(tmpFilePahtLineEdit)
        {
            QString filePath = QFileDialog::getOpenFileName(0, "Open " + prop.getName() + " file", FileSystemInterface::getInstance().getHipnosHome().path(), "CSV Files (*.csv)");
            if(!filePath.isEmpty() && !filePath.isNull())
            {
                tmpFilePahtLineEdit->setText(filePath);
            }
        }
    }

    /**
     * @brief Internal slot used to catch a QSpinBox::valueChanged() signal
     *
     * @param v The new value
     */
    void onSpinBoxChanged(int v)
    {
        prop.setValue(v);
        prop.getComponent()->enqueueProperty(prop);
        prop.getComponent()->setChanged();
        emit propertyChanged(prop);
    }

    /**
     * @brief Internal slot used to catch a QDoubleSpinBox::valueChanged() signal
     *
     * @param v The new value
     */
    void onSpinBoxChanged(double v)
    {
        prop.setValue(v);
        prop.getComponent()->enqueueProperty(prop);
        prop.getComponent()->setChanged();
        emit propertyChanged(prop);
    }

    /**
     * @brief Internal slot used to catch a QComboBox::activated() signal
     *
     * @param o The new value
     */
    void onComboBoxChanged(QString o)
    {
        prop.setSelectedValue(o);
        prop.getComponent()->enqueueProperty(prop);
        prop.getComponent()->setChanged();
        emit propertyChanged(prop);
    }

    /**
     * @brief Internal slot used to open a SpectrumDialog
     *
     */
    void onSpectrumEdit()
    {
        SpectrumDialog dia(prop.getSpectrum());
        dia.exec();
        prop.setSpectrum(dia.getSpectrum());
        prop.getComponent()->enqueueProperty(prop);
        prop.getComponent()->setChanged();
        emit propertyChanged(prop);
    }

private:
    PipelineComponent::Property prop;   /**< Current property */
    QLineEdit* tmpFilePahtLineEdit;   /**< reference to the file name, only used if for PipelineComponent::Property::PT_FILE */

};

#endif // PROPERTYCHANGEDHANDLER_H
