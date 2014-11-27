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

#include "infobutton.h"
#include <QDebug>

InfoButton::InfoButton(const QString &title, const QString& message, QWidget *parent) :
    QPushButton(parent)
{
    const int buttonSize = 14;
    setIcon(QIcon(":/icons/info.png"));
    setIconSize(QSize(buttonSize-2, buttonSize-2));
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    setFixedSize(buttonSize, buttonSize);
    setStyleSheet("border: none;");
    setFocusPolicy(Qt::NoFocus);
    connect(this, SIGNAL(clicked()), this, SLOT(showTooltip()));
    setCursor(Qt::PointingHandCursor);
    tip = new BalloonTip(title, message);
}

InfoButton::~InfoButton()
{
    delete tip;
}

void InfoButton::showTooltip()
{
    QPoint pos = mapToGlobal(QPoint(size().width()/2, size().height()/2));
    tip->showBalloon(pos, true);
}
