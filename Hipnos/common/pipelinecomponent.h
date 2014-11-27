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

#ifndef PIPELINECOMPONET_H
#define PIPELINECOMPONET_H

#define _USE_MATH_DEFINES
#include <cmath>
#include "pipelinedata.h"
#include "pipelineconnection.h"
#include "threadsafequeue.h"
#include "spectrum.h"

#include <QIcon>
#include <QVariant>
#include <QValidator>
#include <QFile>

/**
 * @brief This is the interface for all implemented optical komponents
 *
 */
class PipelineComponent
{
public:

    /**
     * @brief This class represents a specific property for a component. It acts
     *      more or less as a c++ union or QVariant
     */
    class Property
    {
    public:
        /**
         * @brief Enumeration specifying the property types.
         *
         * Types:
         * - PipelineComponent::Property::PT_STRING a QString editable through a QLineEdit
         * - PipelineComponent::Property::PT_INT a int editable through a QSpinBox
         * - PipelineComponent::Property::PT_DOUBLE a double editable through a QDoubleSpinBox
         * - PipelineComponent::Property::PT_LIST a QList<QString> editable through a QComboBox
         * - PipelineComponent::Property::PT_FILE a QFile editable through a QFileDialog
         * - PipelineComponent::Property::PT_SPECTRUM a Spectrum editable through a SpectrumDialog
         */
        enum PropertyType
        {
            PT_STRING,
            PT_INT,
            PT_DOUBLE,
            PT_LIST,
            PT_FILE,
            PT_SPECTRUM
        };

    private:
        QString name;  
        QString description;  
        QVariant value;  
        QString unit;  
        QVariant minValue;  
        QVariant maxValue;  
        QList<QString> options;  
        PipelineComponent* component;  
        PropertyType type;  
        QValidator* validator;  
        QVariant singleStep;  

    public:
        /**
         * @brief Constructs a PipelineComponent::Property of type PipelineComponent::Property::PT_INT
         *
         * @param c A pointer to the PipelineComponent
         * @param n The name
         * @param d The description
         * @param v The value
         * @param min The minimun allowed value
         * @param max The maximum allowed value
         * @param step The spin step used by te QSpinBox
         */
        Property(PipelineComponent* c, QString n, QString d, int v, int min, int max, int step = 1)
        {
            name = n;
            description = d;
            component = c;
            validator = 0;
            value = QVariant(v);
            minValue = QVariant(min);
            maxValue = QVariant(max);
            singleStep = QVariant(step);
            type = PT_INT;
        }

        /**
         * @brief Constructs a PipelineComponent::Property of type PipelineComponent::Property::PT_DOUBLE
         *
         * @param c A pointer to the PipelineComponent
         * @param n The name
         * @param d The description
         * @param v The value
         * @param min The minimum allowed value
         * @param max The maximum allowed value
         * @param u A QString representing the unit of the value
         * @param step The spin step used by the QDoubleSpinBox
         */
        Property(PipelineComponent* c, QString n, QString d, double v, double min, double max,  QString u, double step = 0.01)
        {
            name = n;
            description = d;
            component = c;
            validator = 0;
            unit = u;
            value = QVariant(v);
            minValue = QVariant(min);
            maxValue = QVariant(max);
            singleStep = QVariant(step);
            type = PT_DOUBLE;
        }

        /**
         * @brief Constructs a PipelineComponent::Property of type PipelineComponent::Property::PT_STRING
         *
         * @param c A pointer to the PipelineComponent
         * @param n The name
         * @param d The description
         * @param v The value
         */
        Property(PipelineComponent* c, QString n, QString d, QString v)
        {
            name = n;
            description = d;
            component = c;
            validator = 0;
            value = QVariant(v);
            type = PT_STRING;
        }

        /**
         * @brief Constructs a PipelineComponent::Property of type PipelineComponent::Property::PT_LIST
         *
         * @param c A pointer to the PipelineComponent
         * @param n The name
         * @param d The description
         * @param v The current value
         * @param o A list of options
         */
        Property(PipelineComponent* c, QString n, QString d, QString v, QList<QString> o)
        {
            name = n;
            description = d;
            component = c;
            validator = 0;
            value = QVariant(v);
            options = o;
            type = PT_LIST;
        }

        /**
         * @brief Constructs a PipelineComponent::Property of type PipelineComponent::Property::PT_FILE
         *
         * @param c A pointer to the PipelineComponent
         * @param n The name
         * @param d The description
         * @param v The referenced file
         */
        Property(PipelineComponent *c, QString n, QString d, const QFile& v)
        {
            name = n;
            description = d;
            component = c;
            validator = 0;
            value = QVariant(v.fileName());
            type = PT_FILE;
        }

        /**
         * @brief Constructs a PipelineComponent::Property of type PipelineComponent::Property::PT_SPECTRUM
         *
         * @param c A pointer to the PipelineComponent
         * @param n The name
         * @param d The description
         * @param s The current Spectrum
         */
        Property(PipelineComponent *c, QString n, QString d, Spectrum s)
        {
            name = n;
            description = d;
            component = c;
            validator = 0;
            value.setValue(s);
            type = PT_SPECTRUM;
        }

        /**
         * @brief Gets a pointer to the PipelineComponent
         *
         * @return PipelineComponent *
         */
        PipelineComponent* getComponent()
        {
            return component;
        }

        /**
         * @brief Sets the PipelineComponent
         *
         * @param c
         */
        void setComponent(PipelineComponent* c)
        {
            component = c;
        }

        /**
         * @brief Gets the name
         *
         * @return QString
         */
        QString getName()
        {
            return name;
        }

        /**
         * @brief Gets the description
         *
         * @return QString
         */
        QString getDescription()
        {
            return description;
        }

        /**
         * @brief Gets the unit
         *
         * @return QString
         */
        QString getUnit()
        {
            return unit;
        }

        /**
         * @brief Sets the name
         *
         * @param n
         */
        void setName(QString n)
        {
            name = n;
        }

        /**
         * @brief Gets the PipelineComponent::Property::PropertyType
         *
         * @return PropertyType
         */
        PropertyType getType()
        {
            return type;
        }

        /**
         * @brief Converts the current value to a QString a returns it
         *
         * @return QString
         */
        QString currentValueToString()
        {
            if(getType() == PT_SPECTRUM)
            {
                QString s;
                foreach(Spectrum::SpectrumEntry e, getSpectrum().getEntries())
                {
                    s += QString::number(e.Frequency) + "," + QString::number(e.Intensity) + "|";
                }
                s.chop(1);
                return s;
            }
            else
            {
                return value.toString();
            }
        }

        /**
         * @brief Tries parse the property value from a given string
         *
         * @param s The string
         * @return bool Returns true on success, otherwise false
         */
        bool currentValueFromString(QString s)
        {
            bool success = true;
            if(getType() == PT_DOUBLE)
            {
                setValue(s.toDouble(&success));
            }
            else if(getType() == PT_INT)
            {
                setValue(s.toInt(&success));
            }
            else if(getType() == PT_STRING)
            {
                setValue(s);
            }
            else if(getType() == PT_LIST)
            {
                if(getPropertyOptions().contains(s))
                {
                    setSelectedValue(s);
                }
                else
                {
                    success = false;
                }
            }
            else if(getType() == PT_FILE)
            {
                setFile(QFile(s));
            }
            else if(getType() == PT_SPECTRUM)
            {
                Spectrum spectrum;
                foreach(QString se, s.split("|"))
                {
                    QStringList vals = se.split(",");
                    if(vals.size() == 2)
                    {
                        spectrum.addEntry(vals.at(0).toDouble(), vals.at(1).toDouble());
                    }
                }
                setSpectrum(spectrum);
            }
            return success;
        }

        /**
         * @brief returns the QValidator
         *
         * @return QValidator *
         */
        QValidator* getValidator()
        {
            return validator;
        }

        /**
         * @brief Sets the QValidator for this property
         *
         * @param v
         */
        void setValidator(QValidator* v)
        {
            validator = v;
        }

        /**
         * @brief Returns the step used by the QSpinBox and QDoubleSpinBox.
         *      This method is only allowed to be called for properties of type PipelineComponent::Property::PT_DOUBLE and PipelineComponent::Property::PT_INT
         *
         * @return QVariant
         */
        QVariant getSingleStep()
        {
            Q_ASSERT_X(type == PT_INT || type == PT_DOUBLE, "PipelineComponent::Property::getSingleStep()", "property is not of type PT_INT or PT_DOUBLE");
            return singleStep;
        }

        // PT_STRING methods
        /**
         * @brief Gets the current value for properties of type PipelineComponent::Property::PT_STRING
         *
         * @return QString
         */
        QString getStringValue()
        {
            Q_ASSERT_X(type == PT_STRING, "PipelineComponent::Property::getStringValue()", "property is not of type PT_STRING");
            return value.toString();
        }

        /**
         * @brief Sets the current value for properties of type PipelineComponent::Property::PT_STRING
         *
         * @param v
         */
        void setValue(QString v)
        {
            Q_ASSERT_X(type == PT_STRING, "PipelineComponent::Property::setValue(QString v)", "property is not of type PT_STRING");
            value = QVariant(v);
        }

        // PT_INT methods
        /**
         * @brief Gets the current value for properties of type PipelineComponent::Property::PT_INT
         *
         * @return int
         */
        int getIntValue()
        {
            Q_ASSERT_X(type == PT_INT, "PipelineComponent::Property::getIntValue()", "property is not of type PT_INT");
            return value.toInt();
        }

        /**
         * @brief Sets the current value for properties of type PipelineComponent::Property::PT_INT
         *
         * @param v
         */
        void setValue(int v)
        {
            Q_ASSERT_X(type == PT_INT, "PipelineComponent::Property::setValue(int v)", "property is not of type PT_INT");
            value = QVariant(v);
        }

        /**
         * @brief Gets the minimum allowed value for properties of type PipelineComponent::Property::PT_INT
         *
         * @return int
         */
        int getIntMinValue()
        {
            Q_ASSERT_X(type == PT_INT, "PipelineComponent::Property::getIntMinValue()", "property is not of type PT_INT");
            return minValue.toInt();
        }

        /**
         * @brief Gets the maximum allowed value for properties of type PipelineComponent::Property::PT_INT
         *
         * @return int
         */
        int getIntMaxValue()
        {
            Q_ASSERT_X(type == PT_INT, "PipelineComponent::Property::getIntMaxValue()", "property is not of type PT_INT");
            return maxValue.toInt();
        }

        // PT_DOUBLE methods
        /**
         * @brief Gets the current value for properties of type PipelineComponent::Property::PT_DOUBLE
         *
         * @return double
         */
        double getDoubleValue()
        {
            Q_ASSERT_X(type == PT_DOUBLE, "PipelineComponent::Property::getDoubleValue()", "property is not of type PT_DOUBLE");
            return value.toDouble();
        }

        /**
         * @brief Sets the current value for properties of type PipelineComponent::Property::PT_DOUBLE
         *
         * @param v
         */
        void setValue(double v)
        {
            Q_ASSERT_X(type == PT_DOUBLE, "PipelineComponent::Property::setValue(double v)", "property is not of type PT_DOUBLE");
            value = QVariant(v);
        }

        /**
         * @brief Gets the minimum allowed value for properties of type PipelineComponent::Property::PT_DOUBLE
         *
         * @return double
         */
        double getDoubleMinValue()
        {
            Q_ASSERT_X(type == PT_DOUBLE, "PipelineComponent::Property::getDoubleMinValue()", "property is not of type PT_DOUBLE");
            return minValue.toDouble();
        }

        /**
         * @brief Gets the maximum allowed value for properties of type PipelineComponent::Property::PT_DOUBLE
         *
         * @return double
         */
        double getDoubleMaxValue()
        {
            Q_ASSERT_X(type == PT_DOUBLE, "PipelineComponent::Property::getDoubleMaxValue()", "property is not of type PT_DOUBLE");
            return maxValue.toDouble();
        }

        // PT_LIST
        /**
         * @brief Gets the current value for properties of type PipelineComponent::Property::PT_LIST
         *
         * @return QString
         */
        QString getSelectedValue()
        {
            Q_ASSERT_X(type == PT_LIST, "PipelineComponent::Property::getSelectedValue()", "property is not of type PT_LIST");
            return value.toString();
        }

        /**
         * @brief Sets the current value for properties of type PipelineComponent::Property::PT_LIST
         *
         * @param v NOTE: The option list must contain this value
         */
        void setSelectedValue(QString v)
        {
            Q_ASSERT_X(type == PT_LIST, "PipelineComponent::Property::setSelectedValue(QString v)", "property is not of type PT_LIST");
            Q_ASSERT_X(options.contains(v), "PipelineComponent::Property::setSelectedValue(QString v)", QString("value " + v + " is not a valid option").toStdString().c_str());
            value = QVariant(v);
        }

        /**
         * @brief Gets the option for properties of type PipelineComponent::Property::PT_LIST
         *
         * @return QList<QString>
         */
        QList<QString> getPropertyOptions()
        {
            Q_ASSERT_X(type == PT_LIST, "PipelineComponent::Property::getSelectedValue()", "property is not of type PT_LIST");
            return options;
        }

        //PT_FILE
        /**
         * @brief Sets the current file for properties of type PipelineComponent::Property::PT_FILE
         *
         * @param f
         */
        void setFile(const QFile& f)
        {
            Q_ASSERT_X(type == PT_FILE, "PipelineComponent::Property::setFile()", "property is not of type PT_FILE");
            value = QVariant(f.fileName());
        }

        /**
         * @brief Gets the current file name for properties of type PipelineComponent::Property::PT_FILE
         *
         * @return QString
         */
        QString getFileName()
        {
            Q_ASSERT_X(type == PT_FILE, "PipelineComponent::Property::getFile()", "property is not of type PT_FILE");
            return value.toString();
        }

        //PT_SPECTRUM
        /**
         * @brief Sets the current Spectrum for properties of type PipelineComponent::Property::PT_SPECTRUM
         *
         * @param s
         */
        void setSpectrum(Spectrum s)
        {
            Q_ASSERT_X(type == PT_SPECTRUM, "PipelineComponent::Property::setSpectrum()", "property is not of type PT_SPECTRUM");
            value.setValue(s);
        }

        /**
         * @brief Gets the current Spectrum for properties of type PipelineComponent::Property::PT_SPECTRUM
         *
         * @return Spectrum
         */
        Spectrum getSpectrum()
        {
            Q_ASSERT_X(type == PT_SPECTRUM, "PipelineComponent::Property::getSpectrum()", "property is not of type PT_SPECTRUM");
            return value.value<Spectrum>();
        }

    };

    /**
     * @brief Structure used to compare and sort PipelineComponent::Property by name
     *
     */
    struct LessThan
    {
        bool operator()(PipelineComponent* a, PipelineComponent* b) const
        {
            return (a->getName() < b->getName());
        }
    };

    explicit PipelineComponent();
    virtual ~PipelineComponent();

    void setNumberOfInputConnections(int n);
    void setNumberOfOutputConnections(int n);
    int getNumberOfInputConnections();
    int getNumberOfOutputConnections();
    double getLength();
    QString getName();
    void setName(QString n);
    bool isChanged();
    PipelineConnection* getInputConnection(int i = 0);
    PipelineConnection* getOutputConnection(int i = 0);
    void setInputConnection(PipelineConnection * c, int i = 0);
    void setOutputConnection(PipelineConnection* c, int i = 0);
    void computePropagation(PipelineDataType dataType, double z);
    void enqueueProperty(Property p);

    virtual bool isBasicComponent();
    virtual QString getLabelText();
    virtual QIcon getIcon();
    virtual void setChanged();
    virtual void flush();

    virtual QString getType() = 0;
    virtual PipelineComponent* clone() = 0;
    virtual QList<Property> getProperties() = 0;
    virtual void setProperty(Property p) = 0;

    static bool isNameValid(QString n);
    static QString getUniqueComponentName(QString type);
    static void freeUniqueName(QString n);

protected:
    virtual void gaussPropagation(double z) = 0;
    virtual void fourierPropagation(double z) = 0;

    double length;  
    bool changed;  

private:
    static QSet<QString> uniqueComponentNames;  

    ThreadSafeQueue<Property> queuedProperties;  

    // use getters to acces this members from inherited components
    QString name;  
    QVector<PipelineConnection*> input;  
    QVector<PipelineConnection*> output;  
};

#endif // PIPELINECOMPONET_H
