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

#include "mathplugindialog.h"
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QDebug>

MathPluginDialog::MathPluginDialog(QList<MathPlugin *> plugins) :
    QDialog(0)
{
    setWindowIcon(QIcon(":/icons/app-icon.png"));
    MathPlugin* selectedPlugin = plugins.first();
    uint performanceHint = 0;
    foreach(MathPlugin* p, plugins)
    {
        if(p->getPerformanceHint() >= performanceHint)
        {
            performanceHint = p->getPerformanceHint();
            selectedPlugin = p;
        }
    }

    QVBoxLayout *layout = new QVBoxLayout;

    QLabel* title = new QLabel("<b>Select a Math Plugin form the list:</b>");
    layout->addWidget(title);

    layout->addSpacing(20);

    foreach(MathPlugin* p, plugins)
    {
        QRadioButton* r = new QRadioButton(p->getName());
        layout->addWidget(r);
        if(p == selectedPlugin)
        {
            r->setChecked(true);
        }
        radioButtons[r] = p;
    }

    layout->addSpacing(20);

    okButton = new QPushButton("OK");
    connect(okButton,SIGNAL(clicked()),this,SLOT(accept()));
    layout->addWidget(okButton);


    setLayout(layout);
}

MathPluginDialog::~MathPluginDialog()
{
}

MathPlugin* MathPluginDialog::selectPlugin(QList<MathPlugin *> ps)
{
    if(ps.size() == 1)
    {
        return ps.first();
    }
    MathPluginDialog dia(ps);
    dia.exec();
    MathPlugin* p = dia.getSelectedMathPlugin();
    if(p)
    {
        return p;
    }
    else
    {
        return ps.first();
    }
}

MathPlugin* MathPluginDialog::getSelectedMathPlugin()
{
    foreach(QRadioButton* r, radioButtons.keys())
    {
        if(r->isChecked())
        {
            return radioButtons[r];
        }
    }
    return 0;
}
