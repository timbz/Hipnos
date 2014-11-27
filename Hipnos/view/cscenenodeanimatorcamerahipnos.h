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

//#ifndef CSCENENODEANIMATORCAMERAHIPNOS_H
//#define CSCENENODEANIMATORCAMERAHIPNOS_H

//#include <irrlicht.h>
//#include <ISceneNodeAnimatorCameraFPS.h>
//#include <ICursorControl.h>

//using namespace irr::scene;
//using namespace irr;

//class CSceneNodeAnimatorCameraHipnos : public ISceneNodeAnimatorCameraFPS
//{

//    public:

//        //! Constructor
//        CSceneNodeAnimatorCameraHipnos(gui::ICursorControl* cursorControl,
//                f32 rotateSpeed = 100.0f, f32 moveSpeed = .5f, f32 jumpSpeed=0.f,
//                SKeyMap* keyMapArray=0, u32 keyMapSize=0, bool noVerticalMovement=false,
//                bool invertY=false);

//        //! Destructor
//        virtual ~CSceneNodeAnimatorCameraHipnos();

//        //! Animates the scene node, currently only works on cameras
//        virtual void animateNode(ISceneNode* node, u32 timeMs);

//        //! Event receiver
//        virtual bool OnEvent(const SEvent& event);

//        //! Returns the speed of movement in units per second
//        virtual f32 getMoveSpeed() const;

//        //! Sets the speed of movement in units per second
//        virtual void setMoveSpeed(f32 moveSpeed);

//        //! Returns the rotation speed
//        virtual f32 getRotateSpeed() const;

//        //! Set the rotation speed
//        virtual void setRotateSpeed(f32 rotateSpeed);

//        //! Sets the keyboard mapping for this animator
//        //! \param keymap: an array of keyboard mappings, see SKeyMap
//        //! \param count: the size of the keyboard map array
//        virtual void setKeyMap(SKeyMap *map, u32 count);

//        //! Sets whether vertical movement should be allowed.
//        virtual void setVerticalMovement(bool allow);

//        //! Sets whether the Y axis of the mouse should be inverted.
//        /** If enabled then moving the mouse down will cause
//        the camera to look up. It is disabled by default. */
//        virtual void setInvertMouse(bool invert);

//        //! This animator will receive events when attached to the active camera
//        virtual bool isEventReceiverEnabled() const
//        {
//                return true;
//        }

//        //! Returns the type of this animator
//        virtual ESCENE_NODE_ANIMATOR_TYPE getType() const
//        {
//                return ESNAT_CAMERA_FPS;
//        }

//        //! Creates a clone of this animator.
//        /** Please note that you will have to drop
//        (IReferenceCounted::drop()) the returned pointer once you're
//        done with it. */
//        virtual ISceneNodeAnimator* createClone(ISceneNode* node, ISceneManager* newManager=0);

//        struct SCamKeyMap
//        {
//                SCamKeyMap() {};
//                SCamKeyMap(s32 a, EKEY_CODE k) : action(a), keycode(k) {}

//                s32 action;
//                EKEY_CODE keycode;
//        };

//        //! Sets the keyboard mapping for this animator
//        /** Helper function for the clone method.
//        \param keymap the new keymap array */
//        void setKeyMap(const core::array<SCamKeyMap>& keymap);

//private:
//        void allKeysUp();

//        gui::ICursorControl *CursorControl;

//        f32 MaxVerticalAngle;

//        f32 MoveSpeed;
//        f32 RotateSpeed;
//        f32 JumpSpeed;
//        // -1.0f for inverted mouse, defaults to 1.0f
//        f32 MouseYDirection;

//        s32 LastAnimationTime;

//        core::array<SCamKeyMap> KeyMap;
//        core::position2d<f32> CenterCursor, CursorPos;

//        bool CursorKeys[6];

//        bool firstUpdate;
//        bool NoVerticalMovement;

//};

//#endif // CSCENENODEANIMATORCAMERAHIPNOS_H
