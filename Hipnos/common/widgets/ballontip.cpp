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

#include "ballontip.h"
#include <QLabel>
#include <QPushButton>
#include <QGridLayout>
#include <QPainter>
#include <QApplication>
#include <QDesktopWidget>
#include <QBitmap>
#include <QDebug>

BalloonTip::BalloonTip(const QString& title, const QString& message)
    : QWidget(0,Qt::Popup)
{
    const int maxWidth = 400;

    QLabel *titleLabel = new QLabel;
    titleLabel->installEventFilter(this);
    titleLabel->setText(title);
    QFont f = titleLabel->font();
    f.setBold(true);
    titleLabel->setFont(f);
    titleLabel->setMaximumWidth(maxWidth);

    QLabel *msgLabel = new QLabel;
    msgLabel->installEventFilter(this);
    msgLabel->setText(message);
    msgLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    msgLabel->setMaximumWidth(maxWidth);
    msgLabel->setWordWrap(true);
    //qDebug() << msgLabel->text();

    QGridLayout *layout = new QGridLayout;
    layout->addWidget(titleLabel, 0, 0);
    layout->addWidget(msgLabel, 1, 0);
    layout->setSizeConstraint(QLayout::SetFixedSize);
    layout->setMargin(3);
    setLayout(layout);

    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(0xf0, 0xf0, 0xf0));
    pal.setColor(QPalette::WindowText, Qt::black);
    setPalette(pal);
}

BalloonTip::~BalloonTip()
{
}

void BalloonTip::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.drawPixmap(rect(), pixmap);
}

void BalloonTip::resizeEvent(QResizeEvent *ev)
{
    QWidget::resizeEvent(ev);
}

void BalloonTip::showBalloon(const QPoint& pos, bool showArrow)
{
    QRect scr = QApplication::desktop()->screenGeometry(pos);
    QSize sh = sizeHint();
    const int border = 1;
    const int ah = 18, ao = 18, aw = 18, rc = 7;
    bool arrowAtTop = (pos.y() < (sh.height() + ah));
    bool arrowAtLeft = (pos.x() + sh.width() - ao < scr.width());
    setContentsMargins(border + 3,  border + (arrowAtTop ? ah : 0) + 2, border + 3, border + (arrowAtTop ? 0 : ah) + 2);
    updateGeometry();
    sh  = sizeHint();

    int ml, mr, mt, mb;
    QSize sz = sizeHint();
    if (!arrowAtTop) {
        ml = mt = 0;
        mr = sz.width() - 1;
        mb = sz.height() - ah - 1;
    } else {
        ml = 0;
        mt = ah;
        mr = sz.width() - 1;
        mb = sz.height() - 1;
    }

    QPainterPath path;
    path.moveTo(ml + rc, mt);
    if (arrowAtTop && arrowAtLeft) {
        if (showArrow) {
            path.lineTo(ml + ao, mt);
            path.lineTo(ml + ao, mt - ah);
            path.lineTo(ml + ao + aw, mt);
        }
        move(qMax(pos.x() - ao, scr.left() + 2), pos.y());
    } else if (arrowAtTop && !arrowAtLeft) {
        if (showArrow) {
            path.lineTo(mr - ao - aw, mt);
            path.lineTo(mr - ao, mt - ah);
            path.lineTo(mr - ao, mt);
        }
        move(qMin(pos.x() - sh.width() + ao, scr.right() - sh.width() - 2), pos.y());
    }
    path.lineTo(mr - rc, mt);
    path.arcTo(QRect(mr - rc*2, mt, rc*2, rc*2), 90, -90);
    path.lineTo(mr, mb - rc);
    path.arcTo(QRect(mr - rc*2, mb - rc*2, rc*2, rc*2), 0, -90);
    if (!arrowAtTop && !arrowAtLeft) {
        if (showArrow) {
            path.lineTo(mr - ao, mb);
            path.lineTo(mr - ao, mb + ah);
            path.lineTo(mr - ao - aw, mb);
        }
        move(qMin(pos.x() - sh.width() + ao, scr.right() - sh.width() - 2),
             pos.y() - sh.height());
    } else if (!arrowAtTop && arrowAtLeft) {
        if (showArrow) {
            path.lineTo(ao + aw, mb);
            path.lineTo(ao, mb + ah);
            path.lineTo(ao, mb);
        }
        move(qMax(pos.x() - ao, scr.x() + 2), pos.y() - sh.height());
    }
    path.lineTo(ml + rc, mb);
    path.arcTo(QRect(ml, mb - rc*2, rc*2, rc*2), -90, -90);
    path.lineTo(ml, mt + rc);
    path.arcTo(QRect(ml, mt, rc*2, rc*2), 180, -90);

    // Set the mask
    QBitmap bitmap = QBitmap(sizeHint());
    bitmap.fill(Qt::color0);
    QPainter painter1(&bitmap);
    painter1.setPen(QPen(Qt::color1, border));
    painter1.setBrush(QBrush(Qt::color1));
    painter1.drawPath(path);
    setMask(bitmap);

    // Draw the border
    pixmap = QPixmap(sz);
    QPainter painter2(&pixmap);
    painter2.setPen(QPen(palette().color(QPalette::Window).darker(160), border));
    painter2.setBrush(palette().color(QPalette::Window));
    painter2.drawPath(path);

    show();
}

void BalloonTip::mousePressEvent(QMouseEvent *e)
{
    close();
}
