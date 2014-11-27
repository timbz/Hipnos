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

#ifndef HIPNOSSETTINGS_H
#define HIPNOSSETTINGS_H

#include <QSettings>
#include <QDir>

static QString HIPNOS_HOME_DIR_NAME = "Hipnos";  

/**
 * @brief This singelton class extends QSettings. It is used to store and load application setting
 *
 */
class HipnosSettings : public QSettings
{

public:

    /**
     * @brief Returns a reference to the singelton instance
     *
     * @return HipnosSettings A reference to the instance
     */
    static HipnosSettings& getInstance()
    {
        static HipnosSettings instance;
        return instance;
    }

    /**
     * @brief returns true if the pipeline cash is enabled, otherwise false
     *
     * @return bool
     */
    bool usePipelieCash()
    {
        return value("pipelineCash", QVariant(true)).toBool();
    }

    /**
     * @brief sets and stores if the application should use cashing in the simulation pipeline
     *
     * @param f
     */
    void setUsePipelineCash(bool f)
    {
        setValue("pipelineCash", QVariant(f));
    }

private:
    /**
     * @brief Private constructor
     *
     */
    HipnosSettings() :
        QSettings(QDir::home().path() + QDir::separator() + HIPNOS_HOME_DIR_NAME + QDir::separator() + "settings.ini",
                  QSettings::IniFormat){}

    /**
     * @brief Not implemented (it' s a singelton)
     *
     * @param
     */
    HipnosSettings(HipnosSettings const&);

    /**
     * @brief Not implemented (it' s a singelton)
     *
     * @param
     */
    void operator=(HipnosSettings const&);

};

#endif // HIPNOSSETTINGS_H
