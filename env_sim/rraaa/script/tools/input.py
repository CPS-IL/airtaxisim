# ==============================================================================
# -- Input -----------------------------------------------------------
# ==============================================================================

import sys
import carla

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_r
    from pygame.locals import K_l
    from pygame.locals import K_o
    from pygame.locals import K_k
    from pygame.locals import K_z
    from pygame.locals import K_c
    from pygame.locals import K_e
    from pygame.locals import K_f

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import rospy


def exit_game():
    """Shuts down program and PyGame"""
    pygame.quit()
    rospy.signal_shutdown('Closed by the user!')
    sys.exit()


class InputControl(object):
    """Class that handles input received such as keyboard and mouse."""

    def __init__(self):
        
        """Initializes input member variables when instance is created."""
        self.mouse_pos = (0, 0)
        self.mouse_offset = [0.0, 0.0]
        self.wheel_offset = 0.1
        self.wheel_amount = 0.025
        
        self._reset_being_asked = False
        self._reset_acknowleged = False

        self._lateral_move_cmd = 0  # Left or Right
        self._longitudinal_move_cmd = 0 # Forward or Backward
        self._vertical_move_cmd = 0 # Up or Down
        self._pitch_rate_cmd = 0
        self._roll_rate_cmd = 0
        self._yaw_rate_cmd = 0 

        self.mouse_pos = (0, 0)
        self.mouse_pos_click = (0,0)
        
        rospy.set_param('reset_ack', False)
    
    def tick(self, clock):
        """Executed each frame. Calls method for parsing input."""
        return self.parse_input(clock)

    def _parse_events(self):
        """Parses input events. These events are executed only once when pressing a key."""
        self.mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_game()
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    exit_game()

        self._reset_acknowleged = rospy.get_param('reset_ack')
        if self._reset_acknowleged:
            self._reset_being_asked = False
            rospy.set_param('reset_called', False)
        else:
            rospy.set_param('reset_called', self._reset_being_asked)
    


    def _parse_keys(self, milliseconds):
        """Parses keyboard input when keys are pressed"""
        keys = pygame.key.get_pressed()

        if keys[K_UP] or keys[K_w]:
            self._longitudinal_move_cmd = -1  # Forward
        elif keys[K_DOWN] or keys[K_s]:
            self._longitudinal_move_cmd =  1 # Backward
        elif keys[K_RIGHT] or keys[K_d]:
            self._lateral_move_cmd = 1   # Right
        elif keys[K_LEFT] or keys[K_a]:
            self._lateral_move_cmd = -1  # Left
        elif keys[K_o]:
            self._vertical_move_cmd = -1   # Up
        elif keys[K_l]:
            self._vertical_move_cmd = 1  # Down
        else:
            self._longitudinal_move_cmd = 0
            self._lateral_move_cmd = 0
            self._vertical_move_cmd = 0


        if keys[K_r]:
            self._pitch_rate_cmd = -1  
        elif keys[K_f]:
            self._pitch_rate_cmd =  1 
        elif keys[K_z]:
            self._roll_rate_cmd =  -1 
        elif keys[K_c]:
            self._roll_rate_cmd = 1 
        elif keys[K_q]:
            self._yaw_rate_cmd =  -1 
        elif keys[K_e]:
            self._yaw_rate_cmd = 1 
        else:
            self._pitch_rate_cmd = 0  
            self._roll_rate_cmd = 0 
            self._yaw_rate_cmd = 0 


        

    def _parse_mouse(self):
        """Parses mouse input"""
        if pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()
            self.mouse_offset[0] += (1.0 / self.wheel_offset) * (x - self.mouse_pos[0])
            self.mouse_offset[1] += (1.0 / self.wheel_offset) * (y - self.mouse_pos[1])
            self.mouse_pos_click = (x, y)

            

    def parse_input(self, clock):
        """Parses the input, which is classified in keyboard events and mouse"""
        self._parse_events()
        self._parse_mouse()
        self._parse_keys(clock.get_time())
        


    @staticmethod
    def _is_quit_shortcut(key):
        """Returns True if one of the specified keys are pressed"""
        return (key == K_ESCAPE)
    
