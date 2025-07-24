import pygame
from tools.constants import COLOR_BLACK, COLOR_WHITE





# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpTextManager(object):
    """
    Use ARROWS or
        WASD keys for control.

        W            : throttle
        S            : brake
        A/D          : steer left/right
        Q            : toggle reverse
        Space        : hand-brake
        P            : toggle autopilot
        M            : toggle manual
                       transmission
        ,/.          : gear up/down
        ESC          : quit
    """
    def __init__(self, display_man, display_pos):

        self.display_man = display_man
        self.display_pos = display_pos

        lines = self.__doc__.split('\n')
        self.font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.line_space = 18
        self.seconds_left = 0
        self.surface = pygame.Surface(self.display_man.get_display_size())
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (5, n * self.line_space))
            self._render = True #False
        self.surface.set_alpha(220)

        self.display_man.add_sensor(self)


    def toggle(self):
        self._render = not self._render

    # def render(self, display):
    #     if self._render:
    #         display.blit(self.surface, self.pos)

    def render(self):

        if self._render:
            if self.surface is not None:
                offset = self.display_man.get_display_offset(self.display_pos)
                self.display_man.display.blit(self.surface, offset)
        else:
            self.surface = pygame.Surface(self.display_man.get_display_size())
            self.surface.fill(COLOR_BLACK)

            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)




class InfoTextManager(object):

    def __init__(self, display_man, display_pos):
        pygame.font.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.surface = None
        self.info_text = None
        self.display_man = display_man
        self.display_pos = display_pos
        self.display_man.add_sensor(self)
    
    def tick(self, df_veh_state, df_world_state, input_control):

        self.info_text = [
            '',
            '   Sever / Client:  % 2.1f / % 2.1f FPS'  % (df_world_state.loc['world']['server_fps'], df_world_state.loc['world']['client_fps']), 
            '',
            '   Attack  (K):  %s  '  % ('On' if input_control._image_attack_enabled else 'Off'),
            '   Autopilot (P):  %s   Tracking  (L):  %s  '  % (('On' if bool(df_world_state.loc['world']['autopilot_on']) else 'Off'), ('On' if bool(df_world_state.loc['world']['tracking_control_on']) else 'Off')),
            '',
            '       Steer :  % 2.1f' % (df_veh_state.loc['ego_vehicle']['steer']*100),
            '       Accel:   % 2.1f' % (df_veh_state.loc['ego_vehicle']['throttle']*100),
            '       Brake:   % 2.1f' % (df_veh_state.loc['ego_vehicle']['brake']*100),
            '',
            '   Gearbox  (M):  %s'     % ('Manual' if bool(df_veh_state.loc['ego_vehicle']['manual_gear_shift']) else 'Automatic'),  
            '   Grear   (,/.) :  %f'     % (int(df_veh_state.loc['ego_vehicle']['gear'])),
            '']
        
        self.surface = pygame.Surface(self.display_man.get_display_size())
        self.surface.fill(COLOR_BLACK)

        h_offset, v_offset = (0, 0)
        for item in self.info_text:
            text_surface = self.font.render(item, True, COLOR_WHITE)
            v_offset += 18
            self.surface.blit(text_surface, (h_offset, v_offset))

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)
        else:
            self.surface = pygame.Surface(self.display_man.get_display_size())
            self.surface.fill(COLOR_BLACK)

            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)