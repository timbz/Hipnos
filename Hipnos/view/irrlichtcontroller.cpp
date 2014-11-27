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

//#include "irrlichtcontroller.h"


//using namespace irr;

//IrrlichtController::IrrlichtController(/*QIrrlichtWidget* widget,*/ QObject *parent) :
//    QObject(parent)
//{
//    /*
//    //m_widget = widget;
//    if(widget == NULL)
//    {
//        m_device = 0;
//        // exception
//        return;
//    }
//    // Initialize the widget
//    m_widget->init();

//    // attach update slot
//    connect( m_widget, SIGNAL(update(irr::u32)),
//             this, SLOT(update(irr::u32)));

//    // save pointers
//    m_device = m_widget->getIrrlichtDevice();
//    m_guienv = m_device->getGUIEnvironment();
//    m_manager = m_device->getSceneManager();

//    // init Camera
//    scene::ICameraSceneNode* cam = m_manager->addCameraSceneNode();
//    cam->setPosition( core::vector3df(100,0,0) );
//    cam->setTarget( core::vector3df(0,0,0) );

//    // add input controller
//    m_inputController = new InputController(cam, m_device);
//    m_device->setEventReceiver(m_inputController);

//    // test scene
//    m_guienv->addStaticText(L"Hello World!",  irr::core::rect<s32>(10,10,260,22), true);

//    // Create a small box
//    scene::ISceneNode* node = m_manager->addCubeSceneNode();
//    node->setMaterialFlag( video::EMF_LIGHTING, false );

//    scene::ISceneNodeAnimator* anim = m_manager->createFlyCircleAnimator( core::vector3df(0,0,0), 20 );
//    //node->addAnimator(anim);
//    anim->drop();
//    */
//}

//void IrrlichtController::update(irr::u32 timeMs)
//{
//    m_inputController->update(timeMs);
//}
