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

#ifndef MATHPLUGINDIALOG_H
#define MATHPLUGINDIALOG_H

#include <QDialog>
#include <QRadioButton>
#include <QHash>

#include "common/math/mathplugin.h"

/**
 * @brief QDialog used to display a list of available MathPlugin
 *
 */
class MathPluginDialog : public QDialog
{
    Q_OBJECT

public:
    MathPluginDialog(QList<MathPlugin*> ps);
    ~MathPluginDialog();

    MathPlugin* getSelectedMathPlugin();

    static MathPlugin* selectPlugin(QList<MathPlugin*> ps);
    
private:
     QPushButton* okButton;
     QHash<QRadioButton*, MathPlugin*> radioButtons;
};

#endif // MATHPLUGINDIALOG_H
