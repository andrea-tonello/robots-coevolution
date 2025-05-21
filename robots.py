import random
import math

class Robot:
    """
    Defines a robot agent in a 2D arena. 

    Attributes
    - x, y: position
    - health: starts at 100, decreases when hit
    - ammo: starts at 50, consumed by shooting
    - direction: angle in radians.
    - sensors: dictionary of input values for decision-making.

    Methods
    - update_sensors(opponent, arena_size): calculates
        - Distance and direction to enemy.
        - Wall proximity.
        - Own health and ammo.
    - execute_action(action, opponent, arena_size):
        - Applies either movement, turning, shooting, reloading, or does nothing.
    """

    def __init__(self, x, y, health=100, ammo=50):
        self.x = x
        self.y = y
        self.health = health
        self.ammo = ammo
        self.direction = random.uniform(0, 2 * math.pi) # the facing is random
        self.last_action = None
        self.sensors = {
            'enemy_distance': 0,
            'enemy_direction': 0,
            'health': 100,
            'ammo': 50,
            'wall_distance': 0
        }
    
    def update_sensors(self, opponent, arena_size):
        # Distance to opponent sensor
        dx = opponent.x - self.x
        dy = opponent.y - self.y
        self.sensors['enemy_distance'] = math.sqrt(dx**2 + dy**2)
        
        # Direction to opponent sensor (relative to current direction)
        enemy_dir = math.atan2(dy, dx)
        self.sensors['enemy_direction'] = (enemy_dir - self.direction) % (2 * math.pi)
        
        # Health and ammo update
        self.sensors['health'] = self.health
        self.sensors['ammo'] = self.ammo
        
        # Distance to nearest wall
        self.sensors['wall_distance'] = min(
            self.x, arena_size - self.x, 
            self.y, arena_size - self.y
        )
    
    def execute_action(self, action, opponent, arena_size):

        self.last_action = action
        
        if action == 'move_forward':
            move_dist = 5
            new_x = self.x + move_dist * math.cos(self.direction)
            new_y = self.y + move_dist * math.sin(self.direction)
            
            # Check for arena boundaries when moving
            if 0 <= new_x <= arena_size and 0 <= new_y <= arena_size:
                self.x = new_x
                self.y = new_y
        
        # Turn left/right by 22.5 degrees
        elif action == 'turn_left':
            self.direction = (self.direction - math.pi/8) % (2 * math.pi)
        
        elif action == 'turn_right':
            self.direction = (self.direction + math.pi/8) % (2 * math.pi)
        
        # Shoot and check if the shot hit the opponent
        elif action == 'shoot' and self.ammo > 0:
            self.ammo -= 1
            
            dx = opponent.x - self.x
            dy = opponent.y - self.y

            # (the angle is given by atan(dy/dx))
            angle_to_opponent = math.atan2(dy, dx)      # atan2(dy,dx) == atan(dy/dx)
            angle_diff = abs((angle_to_opponent - self.direction) % (2 * math.pi))
            
            # The bullet has a shotgun-like spread (22.5 deg) and a maximum travel distance of 50
            if angle_diff < math.pi/8 and math.sqrt(dx**2 + dy**2) < 50:
                opponent.health -= 20
        
        elif action == 'reload':
            self.ammo = min(50, self.ammo + 10)
        
        elif action == 'do_nothing':
            pass