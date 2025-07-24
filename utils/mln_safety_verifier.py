from enum import Enum, auto
import numpy as np
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional, Union
import math
import logging
import rospy  # Add ROS import

class Action(Enum):
    # Basic Movement
    ACCELERATE = auto()
    DECELERATE = auto()
    MAINTAIN = auto()
    STOP = auto()
    EMERGENCY_STOP = auto()
    
    # Directional
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    DOWN = auto()
    FORWARD = auto()
    BACKWARD = auto()
    
    # Combined Movement
    HOVER_IN_PLACE = auto()
    SLOW_DESCENT = auto()
    RAPID_DESCENT = auto()
    GENTLE_TURN = auto()
    STABILIZE_POSITION = auto()
    
    # Emergency Actions
    EMERGENCY_LANDING_REQUIRED = auto()

class PlanningAction(Enum):
    ACCELERATE = auto()
    DECELERATE = auto()
    MAINTAIN = auto()
    STOP = auto()
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    DOWN = auto()
    FORWARD = auto()
    BACKWARD = auto()

class EnvironmentalPredicate(Enum):
    # Flight Phase
    LANDING = auto()
    FINAL_APPROACH = auto()
    TOUCHDOWN_PHASE = auto()
    TAKEOFF_PHASE = auto()
    CRUISE_PHASE = auto()
    HOVER_PHASE = auto()
    
    # Altitude
    ALTITUDE_ABOVE_100 = auto()
    ALTITUDE_BELOW_100 = auto()
    ALTITUDE_BELOW_50 = auto()
    ALTITUDE_BELOW_20 = auto()
    ALTITUDE_BELOW_10 = auto()
    ALTITUDE_BELOW_5 = auto()
    ALTITUDE_CHANGE_RATE_HIGH = auto()
    ALTITUDE_STABLE = auto()
    
    # Velocity
    VELOCITY_ABOVE_5 = auto()
    VELOCITY_BETWEEN_2_5 = auto()
    VELOCITY_BELOW_2 = auto()
    VELOCITY_BELOW_1 = auto()
    VERTICAL_SPEED_HIGH = auto()
    VERTICAL_SPEED_LOW = auto()
    HORIZONTAL_SPEED_HIGH = auto()
    HORIZONTAL_SPEED_LOW = auto()
    ACCELERATION_HIGH = auto()
    DECELERATION_HIGH = auto()
    SPEED_STABLE = auto()
    
    # Obstacles
    OBSTACLES_IN_5M_RADIUS = auto()
    OBSTACLES_IN_2M_RADIUS = auto()
    OBSTACLES_IN_1M_RADIUS = auto()
    OBSTACLE_BELOW = auto()
    OBSTACLE_ABOVE = auto()
    OBSTACLE_AHEAD_1M = auto()
    OBSTACLE_AHEAD_2M = auto()
    OBSTACLE_DENSITY_HIGH = auto()
    OBSTACLE_MOVING = auto()
    OBSTACLE_ON_PATH = auto()
    PATH_BLOCKED = auto()
    
    # Vehicle State
    YAW_RATE_HIGH = auto()
    ROLL_RATE_HIGH = auto()
    PITCH_RATE_HIGH = auto()
    ATTITUDE_STABLE = auto()
    BATTERY_LOW = auto()
    BATTERY_CRITICAL = auto()
    BATTERY_SUFFICIENT = auto()
    
    # Path and Navigation
    DEVIATION_FROM_PATH_HIGH = auto()
    DEVIATION_FROM_PATH_LOW = auto()
    WAYPOINT_NEAR = auto()
    WAYPOINT_FAR = auto()
    
    # Weather
    WIND_STRONG = auto()
    WIND_GUST_DETECTED = auto()
    VISIBILITY_LOW = auto()
    TURBULENCE_HIGH = auto()
    
    # System Health
    GPS_SIGNAL_STRONG = auto()
    GPS_SIGNAL_WEAK = auto()
    TRAFFIC_ON_COLLISION_COURSE = auto()
    
    # Landing Site
    LANDING_SURFACE_FLAT = auto()
    LANDING_ZONE_CLEAR = auto()
    LANDING_SITE_VERIFIED = auto()
    
    # Passenger
    PASSENGER_COMFORT_PRIORITY = auto()
    
    # System Mode
    ENERGY_EFFICIENT_MODE = auto()

class LogicalOperator(Enum):
    AND = auto()    # Conjunction (∧)
    OR = auto()     # Disjunction (∨)
    NOT = auto()    # Negation (¬)
    IMPLIES = auto() # Implication (→)
    XOR = auto()    # Exclusive OR (⊕)
    FORALL = auto() # Universal Quantifier (∀)
    EXISTS = auto() # Existential Quantifier (∃)
    NAND = auto()   # Not AND (↑)
    NOR = auto()    # Not OR (↓)

@dataclass
class Variable:
    """Represents a variable in quantified expressions"""
    name: str
    domain: str  # e.g., 'Obstacles', 'Waypoints'

@dataclass
class LogicalExpression:
    """Represents a logical expression in first-order logic"""
    operator: LogicalOperator
    predicates: List[Union['LogicalExpression', EnvironmentalPredicate]]
    variables: Optional[List[Variable]] = None  # For quantified expressions

@dataclass
class Rule:
    """Enhanced rule class supporting complex logical expressions"""
    conditions: LogicalExpression
    output_actions: List[Action]
    weight: float = 1.0
    description: str = ""  # Human-readable description of the rule

@dataclass
class SafetyVerificationResult:
    """Class to hold the results of safety verification."""
    verified_actions: Set[Action]
    score: float
    active_predicates: Set[EnvironmentalPredicate]
    triggered_rules: List[Rule]
    is_safe: bool
    safety_warnings: List[str] = field(default_factory=list)

class SafetyVerificationError(Exception):
    """Custom exception for safety verification errors."""
    pass

class ROSLogHandler(logging.Handler):
    """Custom logging handler that forwards logs to ROS logging system."""
    def emit(self, record):
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                rospy.logerr(msg)
            elif record.levelno >= logging.WARNING:
                rospy.logwarn(msg)
            elif record.levelno >= logging.INFO:
                rospy.loginfo(msg)
            else:
                rospy.logdebug(msg)
        except Exception:
            self.handleError(record)

class MLNSafetyVerifier:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the MLN Safety Verifier."""
        try:
            self.config = config or self._get_default_config()
            self.rules = self._initialize_rules()
            self.mutually_exclusive_actions = self._initialize_mutual_exclusions()
            
            # Initialize logger
            self.logger = logging.getLogger("MLNSafetyVerifier")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            
            # Add ROS logging handler
            self.logger.addHandler(ROSLogHandler())
            
            # Validate configuration
            self._validate_config()
        except Exception as e:
            rospy.logerr(f"Failed to initialize safety verifier: {str(e)}")
            raise SafetyVerificationError(f"Failed to initialize safety verifier: {str(e)}")

    def _validate_config(self):
        """Validate configuration parameters."""
        required_keys = {
            'altitude_thresholds', 
            'velocity_thresholds', 
            'attitude_thresholds',
            'safety_distances',
            'battery_thresholds'
        }
        
        if not all(key in self.config for key in required_keys):
            raise SafetyVerificationError(f"Missing required configuration keys: {required_keys - set(self.config.keys())}")

    def _get_default_config(self) -> Dict:
        """Get default configuration values."""
        return {
            'altitude_thresholds': {
                'change_rate_high': 2.0,
                'stable': 0.1
            },
            'velocity_thresholds': {
                'vertical_high': 2.0,
                'vertical_low': 0.5,
                'horizontal_high': 3.0,
                'horizontal_low': 1.0
            },
            'attitude_thresholds': {
                'rate_high': 0.5,
                'rate_stable': 0.1
            },
            'safety_distances': {
                'obstacle_critical': 1.0,
                'obstacle_warning': 2.0,
                'obstacle_caution': 5.0
            },
            'battery_thresholds': {
                'critical': 10,
                'low': 20,
                'sufficient': 30
            }
        }

    def evaluate_environmental_predicates(self, state_data: Dict) -> Set[EnvironmentalPredicate]:
        """Evaluate all environmental predicates based on current state data."""
        active_predicates = set()
        
        # Flight Phase Predicates
        if state_data.get('is_landing', False):
            active_predicates.add(EnvironmentalPredicate.LANDING)
        if state_data.get('is_final_approach', False):
            active_predicates.add(EnvironmentalPredicate.FINAL_APPROACH)
        if state_data.get('is_touchdown', False):
            active_predicates.add(EnvironmentalPredicate.TOUCHDOWN_PHASE)
        if state_data.get('is_takeoff', False):
            active_predicates.add(EnvironmentalPredicate.TAKEOFF_PHASE)
        if state_data.get('is_cruising', False):
            active_predicates.add(EnvironmentalPredicate.CRUISE_PHASE)
        if state_data.get('is_hovering', False):
            active_predicates.add(EnvironmentalPredicate.HOVER_PHASE)

        # Altitude Predicates
        altitude = state_data.get('altitude', 0)
        altitude_rate = state_data.get('altitude_rate', 0)
        if altitude > 100:
            active_predicates.add(EnvironmentalPredicate.ALTITUDE_ABOVE_100)
        if altitude < 100:
            active_predicates.add(EnvironmentalPredicate.ALTITUDE_BELOW_100)
        if altitude < 50:
            active_predicates.add(EnvironmentalPredicate.ALTITUDE_BELOW_50)
        if altitude < 20:
            active_predicates.add(EnvironmentalPredicate.ALTITUDE_BELOW_20)
        if altitude < 10:
            active_predicates.add(EnvironmentalPredicate.ALTITUDE_BELOW_10)
        if altitude < 5:
            active_predicates.add(EnvironmentalPredicate.ALTITUDE_BELOW_5)
        if abs(altitude_rate) > 2.0:
            active_predicates.add(EnvironmentalPredicate.ALTITUDE_CHANGE_RATE_HIGH)
        if abs(altitude_rate) < 0.1:
            active_predicates.add(EnvironmentalPredicate.ALTITUDE_STABLE)

        # Velocity Predicates
        velocity = state_data.get('velocity', 0)
        vertical_speed = state_data.get('vertical_speed', 0)
        horizontal_speed = state_data.get('horizontal_speed', 0)
        acceleration = state_data.get('acceleration', 0)
        
        if velocity > 5:
            active_predicates.add(EnvironmentalPredicate.VELOCITY_ABOVE_5)
        if 2 < velocity < 5:
            active_predicates.add(EnvironmentalPredicate.VELOCITY_BETWEEN_2_5)
        if velocity < 2:
            active_predicates.add(EnvironmentalPredicate.VELOCITY_BELOW_2)
        if velocity < 1:
            active_predicates.add(EnvironmentalPredicate.VELOCITY_BELOW_1)
        if abs(vertical_speed) > 2:
            active_predicates.add(EnvironmentalPredicate.VERTICAL_SPEED_HIGH)
        if abs(vertical_speed) < 0.5:
            active_predicates.add(EnvironmentalPredicate.VERTICAL_SPEED_LOW)
        if horizontal_speed > 3:
            active_predicates.add(EnvironmentalPredicate.HORIZONTAL_SPEED_HIGH)
        if horizontal_speed < 1:
            active_predicates.add(EnvironmentalPredicate.HORIZONTAL_SPEED_LOW)
        if acceleration > 2:
            active_predicates.add(EnvironmentalPredicate.ACCELERATION_HIGH)
        if acceleration < -2:
            active_predicates.add(EnvironmentalPredicate.DECELERATION_HIGH)
        if abs(acceleration) < 0.2:
            active_predicates.add(EnvironmentalPredicate.SPEED_STABLE)

        # Obstacle Predicates
        nearest_obstacle = state_data.get('nearest_obstacle_distance', float('inf'))
        obstacle_density = state_data.get('obstacle_density', 0)
        if nearest_obstacle < 5:
            active_predicates.add(EnvironmentalPredicate.OBSTACLES_IN_5M_RADIUS)
        if nearest_obstacle < 2:
            active_predicates.add(EnvironmentalPredicate.OBSTACLES_IN_2M_RADIUS)
        if nearest_obstacle < 1:
            active_predicates.add(EnvironmentalPredicate.OBSTACLES_IN_1M_RADIUS)
        if state_data.get('obstacle_below', False):
            active_predicates.add(EnvironmentalPredicate.OBSTACLE_BELOW)
        if state_data.get('obstacle_above', False):
            active_predicates.add(EnvironmentalPredicate.OBSTACLE_ABOVE)
        if state_data.get('obstacle_ahead_1m', False):
            active_predicates.add(EnvironmentalPredicate.OBSTACLE_AHEAD_1M)
        if state_data.get('obstacle_ahead_2m', False):
            active_predicates.add(EnvironmentalPredicate.OBSTACLE_AHEAD_2M)
        if obstacle_density > 0.5:
            active_predicates.add(EnvironmentalPredicate.OBSTACLE_DENSITY_HIGH)
        if state_data.get('obstacle_moving', False):
            active_predicates.add(EnvironmentalPredicate.OBSTACLE_MOVING)
        if state_data.get('obstacle_on_path', False):
            active_predicates.add(EnvironmentalPredicate.OBSTACLE_ON_PATH)

        # Vehicle State Predicates
        yaw_rate = abs(state_data.get('yaw_rate', 0))
        roll_rate = abs(state_data.get('roll_rate', 0))
        pitch_rate = abs(state_data.get('pitch_rate', 0))
        if yaw_rate > 0.5:
            active_predicates.add(EnvironmentalPredicate.YAW_RATE_HIGH)
        if roll_rate > 0.5:
            active_predicates.add(EnvironmentalPredicate.ROLL_RATE_HIGH)
        if pitch_rate > 0.5:
            active_predicates.add(EnvironmentalPredicate.PITCH_RATE_HIGH)
        if max(yaw_rate, roll_rate, pitch_rate) < 0.1:
            active_predicates.add(EnvironmentalPredicate.ATTITUDE_STABLE)
        
        battery_level = state_data.get('battery_level', 100)
        if battery_level < 20:
            active_predicates.add(EnvironmentalPredicate.BATTERY_LOW)
        if battery_level < 10:
            active_predicates.add(EnvironmentalPredicate.BATTERY_CRITICAL)
        if battery_level > 30:
            active_predicates.add(EnvironmentalPredicate.BATTERY_SUFFICIENT)

        # Path and Navigation Predicates
        path_deviation = state_data.get('path_deviation', 0)
        if path_deviation > 2:
            active_predicates.add(EnvironmentalPredicate.DEVIATION_FROM_PATH_HIGH)
        if path_deviation < 0.5:
            active_predicates.add(EnvironmentalPredicate.DEVIATION_FROM_PATH_LOW)
        
        waypoint_distance = state_data.get('waypoint_distance', float('inf'))
        if waypoint_distance < 2:
            active_predicates.add(EnvironmentalPredicate.WAYPOINT_NEAR)
        if waypoint_distance > 10:
            active_predicates.add(EnvironmentalPredicate.WAYPOINT_FAR)

        # Weather Predicates
        wind_speed = state_data.get('wind_speed', 0)
        if wind_speed > 5:
            active_predicates.add(EnvironmentalPredicate.WIND_STRONG)
        if state_data.get('wind_gust_detected', False):
            active_predicates.add(EnvironmentalPredicate.WIND_GUST_DETECTED)
        if state_data.get('visibility', 100) < 50:
            active_predicates.add(EnvironmentalPredicate.VISIBILITY_LOW)

        # System Health Predicates
        gps_strength = state_data.get('gps_strength', 1.0)
        if gps_strength > 0.8:
            active_predicates.add(EnvironmentalPredicate.GPS_SIGNAL_STRONG)
        if gps_strength < 0.4:
            active_predicates.add(EnvironmentalPredicate.GPS_SIGNAL_WEAK)

        # Landing Site Predicates
        if state_data.get('landing_surface_flat', False):
            active_predicates.add(EnvironmentalPredicate.LANDING_SURFACE_FLAT)
        if state_data.get('landing_zone_clear', False):
            active_predicates.add(EnvironmentalPredicate.LANDING_ZONE_CLEAR)
        if state_data.get('landing_site_verified', False):
            active_predicates.add(EnvironmentalPredicate.LANDING_SITE_VERIFIED)

        return active_predicates

    def _initialize_rules(self) -> List[Rule]:
        """Initialize all safety rules with their weights."""
        rules = []

        # 0. PLANNING ACTION MAPPING RULES (Priority 10.0-10.5)
        # More conservative actions get higher weights
        rules.extend([
            # Stop - Most conservative (weight 10.5)
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[PlanningAction.STOP]
                ),
                output_actions=[Action.STOP],
                weight=10.5,
                description="Direct mapping from planning stop to actual stop"
            ),
            
            # Decelerate - Very conservative (weight 10.4)
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[PlanningAction.DECELERATE]
                ),
                output_actions=[Action.DECELERATE],
                weight=10.4,
                description="Direct mapping from planning decelerate to actual decelerate"
            ),
            
            # Maintain - Moderately conservative (weight 10.3)
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[PlanningAction.MAINTAIN]
                ),
                output_actions=[Action.MAINTAIN],
                weight=10.3,
                description="Direct mapping from planning maintain to actual maintain"
            ),
            
            # Directional movements - Neutral (weight 10.2)
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[PlanningAction.UP]
                ),
                output_actions=[Action.UP],
                weight=10.2,
                description="Direct mapping from planning up to actual up"
            ),
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[PlanningAction.DOWN]
                ),
                output_actions=[Action.DOWN],
                weight=10.2,
                description="Direct mapping from planning down to actual down"
            ),
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[PlanningAction.LEFT]
                ),
                output_actions=[Action.LEFT],
                weight=10.2,
                description="Direct mapping from planning left to actual left"
            ),
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[PlanningAction.RIGHT]
                ),
                output_actions=[Action.RIGHT],
                weight=10.2,
                description="Direct mapping from planning right to actual right"
            ),
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[PlanningAction.FORWARD]
                ),
                output_actions=[Action.FORWARD],
                weight=10.2,
                description="Direct mapping from planning forward to actual forward"
            ),
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[PlanningAction.BACKWARD]
                ),
                output_actions=[Action.BACKWARD],
                weight=10.2,
                description="Direct mapping from planning backward to actual backward"
            ),
            
            # Accelerate - Least conservative (weight 10.0)
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[PlanningAction.ACCELERATE]
                ),
                output_actions=[Action.ACCELERATE],
                weight=10.0,
                description="Direct mapping from planning accelerate to actual accelerate"
            ),
        ])

        # 1. CRITICAL EMERGENCY RULES (Priority 9.5-10.0) - 6 rules
        rules.extend([
            # Rule 1: Complex obstacle avoidance with quantifier
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.EXISTS,
                    variables=[Variable("obs", "Obstacles")],
                    predicates=[
                        LogicalExpression(
                            operator=LogicalOperator.AND,
                            predicates=[
                                EnvironmentalPredicate.OBSTACLES_IN_1M_RADIUS,
                                EnvironmentalPredicate.VELOCITY_ABOVE_2
                            ]
                        )
                    ]
                ),
                output_actions=[Action.EMERGENCY_STOP],
                weight=10.0,
                description="Emergency stop if any obstacle is too close and speed is high"
            ),

            # Rule 2: Critical battery safety
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.OR,
                    predicates=[
                        EnvironmentalPredicate.BATTERY_CRITICAL,
                        LogicalExpression(
                            operator=LogicalOperator.AND,
                            predicates=[
                                EnvironmentalPredicate.BATTERY_LOW,
                                EnvironmentalPredicate.WAYPOINT_FAR
                            ]
                        )
                    ]
                ),
                output_actions=[Action.EMERGENCY_LANDING_REQUIRED],
                weight=9.8,
                description="Emergency landing for critical battery or low battery far from destination"
            ),

            # Rule 3: Severe weather conditions
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        LogicalExpression(
                            operator=LogicalOperator.OR,
                            predicates=[
                                EnvironmentalPredicate.WIND_STRONG,
                                EnvironmentalPredicate.WIND_GUST_DETECTED
                            ]
                        ),
                        LogicalExpression(
                            operator=LogicalOperator.OR,
                            predicates=[
                                EnvironmentalPredicate.ALTITUDE_ABOVE_100,
                                EnvironmentalPredicate.VELOCITY_ABOVE_5
                            ]
                        )
                    ]
                ),
                output_actions=[Action.SLOW_DESCENT, Action.STABILIZE_POSITION],
                weight=9.5,
                description="Careful descent in challenging weather conditions"
            ),

            # Rule 4: Multiple moving obstacles
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        LogicalExpression(
                            operator=LogicalOperator.EXISTS,
                            variables=[Variable("obs1", "Obstacles")],
                            predicates=[EnvironmentalPredicate.OBSTACLES_IN_2M_RADIUS]
                        ),
                        LogicalExpression(
                            operator=LogicalOperator.EXISTS,
                            variables=[Variable("obs2", "Obstacles")],
                            predicates=[EnvironmentalPredicate.OBSTACLE_MOVING]
                        )
                    ]
                ),
                output_actions=[Action.EMERGENCY_STOP, Action.HOVER_IN_PLACE],
                weight=9.9,
                description="Emergency stop when multiple obstacles detected, including moving ones"
            ),

            # Rule 5: Critical system failure
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.GPS_SIGNAL_WEAK,
                        LogicalExpression(
                            operator=LogicalOperator.OR,
                            predicates=[
                                EnvironmentalPredicate.BATTERY_LOW,
                                EnvironmentalPredicate.WIND_STRONG
                            ]
                        )
                    ]
                ),
                output_actions=[Action.EMERGENCY_LANDING_REQUIRED],
                weight=9.7,
                description="Emergency landing when GPS is weak and either battery is low or wind is strong"
            ),

            # Rule 6: Collision course detection
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.TRAFFIC_ON_COLLISION_COURSE,
                        LogicalExpression(
                            operator=LogicalOperator.NOT,
                            predicates=[EnvironmentalPredicate.ALTITUDE_STABLE]
                        )
                    ]
                ),
                output_actions=[Action.EMERGENCY_STOP, Action.HOVER_IN_PLACE],
                weight=9.6,
                description="Emergency stop when on collision course and altitude not stable"
            )
        ])

        # 2. LANDING SAFETY RULES (Priority 8.5-9.4) - 6 rules
        rules.extend([
            # Rule 7: Complex landing condition
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.LANDING,
                        LogicalExpression(
                            operator=LogicalOperator.NOT,
                            predicates=[
                                LogicalExpression(
                                    operator=LogicalOperator.AND,
                                    predicates=[
                                        EnvironmentalPredicate.LANDING_ZONE_CLEAR,
                                        EnvironmentalPredicate.LANDING_SURFACE_FLAT
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                output_actions=[Action.HOVER_IN_PLACE, Action.STABILIZE_POSITION],
                weight=9.2,
                description="Hold position if landing conditions not optimal"
            ),

            # Rule 8: Final approach safety
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.FINAL_APPROACH,
                        LogicalExpression(
                            operator=LogicalOperator.OR,
                            predicates=[
                                EnvironmentalPredicate.VERTICAL_SPEED_HIGH,
                                EnvironmentalPredicate.HORIZONTAL_SPEED_HIGH
                            ]
                        )
                    ]
                ),
                output_actions=[Action.DECELERATE, Action.STABILIZE_POSITION],
                weight=9.1,
                description="Slow down during final approach if speed too high"
            ),

            # Rule 9: Landing site verification
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.NAND,
                    predicates=[
                        EnvironmentalPredicate.LANDING_SITE_VERIFIED,
                        EnvironmentalPredicate.LANDING_SURFACE_FLAT,
                        EnvironmentalPredicate.LANDING_ZONE_CLEAR
                    ]
                ),
                output_actions=[Action.HOVER_IN_PLACE],
                weight=9.0,
                description="Hold position until landing site fully verified"
            ),

            # Rule 10: Wind conditions during landing
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.LANDING,
                        LogicalExpression(
                            operator=LogicalOperator.OR,
                            predicates=[
                                EnvironmentalPredicate.WIND_STRONG,
                                EnvironmentalPredicate.WIND_GUST_DETECTED
                            ]
                        )
                    ]
                ),
                output_actions=[Action.HOVER_IN_PLACE, Action.STABILIZE_POSITION],
                weight=8.9,
                description="Stabilize in strong winds during landing"
            ),

            # Rule 11: Low visibility landing
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.LANDING,
                        EnvironmentalPredicate.VISIBILITY_LOW,
                        LogicalExpression(
                            operator=LogicalOperator.NOT,
                            predicates=[EnvironmentalPredicate.LANDING_SITE_VERIFIED]
                        )
                    ]
                ),
                output_actions=[Action.HOVER_IN_PLACE],
                weight=8.8,
                description="Hold position in low visibility until landing site verified"
            ),

            # Rule 12: Emergency landing procedure
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.LANDING,
                        EnvironmentalPredicate.BATTERY_LOW,
                        LogicalExpression(
                            operator=LogicalOperator.NOT,
                            predicates=[EnvironmentalPredicate.OBSTACLES_IN_2M_RADIUS]
                        )
                    ]
                ),
                output_actions=[Action.RAPID_DESCENT],
                weight=8.7,
                description="Rapid descent when battery low and no nearby obstacles"
            )
        ])

        # 3. STABILITY AND CONTROL RULES (Priority 7.5-8.4) - 5 rules
        rules.extend([
            # Rule 13: Complex attitude stability
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.OR,
                    predicates=[
                        LogicalExpression(
                            operator=LogicalOperator.AND,
                            predicates=[
                                EnvironmentalPredicate.YAW_RATE_HIGH,
                                EnvironmentalPredicate.ROLL_RATE_HIGH
                            ]
                        ),
                        LogicalExpression(
                            operator=LogicalOperator.AND,
                            predicates=[
                                EnvironmentalPredicate.PITCH_RATE_HIGH,
                                EnvironmentalPredicate.ALTITUDE_CHANGE_RATE_HIGH
                            ]
                        )
                    ]
                ),
                output_actions=[Action.STABILIZE_POSITION, Action.HOVER_IN_PLACE],
                weight=8.3,
                description="Stabilize when multiple attitude rates high"
            ),

            # Rule 14: Weather-induced stability
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.WIND_STRONG,
                        LogicalExpression(
                            operator=LogicalOperator.NOT,
                            predicates=[EnvironmentalPredicate.ALTITUDE_STABLE]
                        )
                    ]
                ),
                output_actions=[Action.STABILIZE_POSITION],
                weight=8.1,
                description="Stabilize in strong winds when altitude not stable"
            ),

            # Rule 15: Turbulence handling
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.TURBULENCE_HIGH,
                        LogicalExpression(
                            operator=LogicalOperator.OR,
                            predicates=[
                                EnvironmentalPredicate.ALTITUDE_ABOVE_100,
                                EnvironmentalPredicate.VELOCITY_ABOVE_5
                            ]
                        )
                    ]
                ),
                output_actions=[Action.DECELERATE, Action.STABILIZE_POSITION],
                weight=8.0,
                description="Reduce speed and stabilize in high turbulence"
            ),

            # Rule 16: GPS signal loss handling
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.GPS_SIGNAL_WEAK,
                        LogicalExpression(
                            operator=LogicalOperator.NOT,
                            predicates=[EnvironmentalPredicate.HOVER_PHASE]
                        )
                    ]
                ),
                output_actions=[Action.HOVER_IN_PLACE, Action.STABILIZE_POSITION],
                weight=7.9,
                description="Hold position when GPS signal weak"
            ),

            # Rule 17: High-speed stability
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.VELOCITY_ABOVE_5,
                        LogicalExpression(
                            operator=LogicalOperator.OR,
                            predicates=[
                                EnvironmentalPredicate.YAW_RATE_HIGH,
                                EnvironmentalPredicate.ROLL_RATE_HIGH
                            ]
                        )
                    ]
                ),
                output_actions=[Action.DECELERATE],
                weight=7.8,
                description="Reduce speed when attitude rates high at high speed"
            )
        ])

        # 4. OBSTACLE AVOIDANCE RULES (Priority 6.5-7.4) - 4 rules
        rules.extend([
            # Rule 18: Dynamic obstacle tracking
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.EXISTS,
                    variables=[Variable("obs", "Obstacles")],
                    predicates=[
                        LogicalExpression(
                            operator=LogicalOperator.AND,
                            predicates=[
                                EnvironmentalPredicate.OBSTACLE_MOVING,
                                EnvironmentalPredicate.OBSTACLES_IN_5M_RADIUS
                            ]
                        )
                    ]
                ),
                output_actions=[Action.HOVER_IN_PLACE, Action.STABILIZE_POSITION],
                weight=7.3,
                description="Hold position when moving obstacles nearby"
            ),

            # Rule 19: Dense obstacle field
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.OBSTACLE_DENSITY_HIGH,
                        LogicalExpression(
                            operator=LogicalOperator.OR,
                            predicates=[
                                EnvironmentalPredicate.VELOCITY_ABOVE_2,
                                EnvironmentalPredicate.VISIBILITY_LOW
                            ]
                        )
                    ]
                ),
                output_actions=[Action.DECELERATE, Action.STABILIZE_POSITION],
                weight=7.1,
                description="Slow down in dense obstacle field"
            ),

            # Rule 20: Path obstruction
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.PATH_BLOCKED,
                        LogicalExpression(
                            operator=LogicalOperator.NOT,
                            predicates=[EnvironmentalPredicate.HOVER_PHASE]
                        )
                    ]
                ),
                output_actions=[Action.HOVER_IN_PLACE],
                weight=7.0,
                description="Hold position when path blocked"
            ),

            # Rule 21: Vertical obstacle avoidance
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.OR,
                    predicates=[
                        EnvironmentalPredicate.OBSTACLE_BELOW,
                        EnvironmentalPredicate.OBSTACLE_ABOVE
                    ]
                ),
                output_actions=[Action.HOVER_IN_PLACE, Action.STABILIZE_POSITION],
                weight=6.9,
                description="Hold position when vertical obstacles detected"
            )
        ])

        # 5. PATH FOLLOWING RULES (Priority 5.5-6.4) - 3 rules
        rules.extend([
            # Rule 22: Path deviation correction
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.DEVIATION_FROM_PATH_HIGH,
                        LogicalExpression(
                            operator=LogicalOperator.NOT,
                            predicates=[EnvironmentalPredicate.OBSTACLES_IN_2M_RADIUS]
                        )
                    ]
                ),
                output_actions=[Action.STABILIZE_POSITION],
                weight=6.3,
                description="Stabilize when deviating from path"
            ),

            # Rule 23: Waypoint approach
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.WAYPOINT_NEAR,
                        EnvironmentalPredicate.VELOCITY_ABOVE_2
                    ]
                ),
                output_actions=[Action.DECELERATE],
                weight=6.1,
                description="Slow down when approaching waypoint"
            ),

            # Rule 24: Efficient cruising
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.CRUISE_PHASE,
                        EnvironmentalPredicate.BATTERY_SUFFICIENT,
                        LogicalExpression(
                            operator=LogicalOperator.NOT,
                            predicates=[
                                LogicalExpression(
                                    operator=LogicalOperator.OR,
                                    predicates=[
                                        EnvironmentalPredicate.OBSTACLES_IN_5M_RADIUS,
                                        EnvironmentalPredicate.WIND_STRONG
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                output_actions=[Action.MAINTAIN],
                weight=5.9,
                description="Maintain cruise when conditions good"
            )
        ])

        # 6. COMFORT AND EFFICIENCY RULES (Priority 4.5-5.4) - 4 rules
        rules.extend([
            # Rule 25: Passenger comfort
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.PASSENGER_COMFORT_PRIORITY,
                        LogicalExpression(
                            operator=LogicalOperator.OR,
                            predicates=[
                                EnvironmentalPredicate.TURBULENCE_HIGH,
                                EnvironmentalPredicate.ACCELERATION_HIGH
                            ]
                        )
                    ]
                ),
                output_actions=[Action.DECELERATE, Action.STABILIZE_POSITION],
                weight=5.3,
                description="Smooth flight for passenger comfort"
            ),

            # Rule 26: Energy efficiency
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.ENERGY_EFFICIENT_MODE,
                        EnvironmentalPredicate.BATTERY_LOW,
                        LogicalExpression(
                            operator=LogicalOperator.NOT,
                            predicates=[EnvironmentalPredicate.LANDING]
                        )
                    ]
                ),
                output_actions=[Action.DECELERATE],
                weight=5.1,
                description="Conserve energy when battery low"
            ),

            # Rule 27: Smooth acceleration
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.ACCELERATION_HIGH,
                        EnvironmentalPredicate.PASSENGER_COMFORT_PRIORITY
                    ]
                ),
                output_actions=[Action.DECELERATE],
                weight=4.9,
                description="Reduce acceleration for passenger comfort"
            ),

            # Rule 28: Efficient altitude maintenance
            Rule(
                conditions=LogicalExpression(
                    operator=LogicalOperator.AND,
                    predicates=[
                        EnvironmentalPredicate.ALTITUDE_STABLE,
                        EnvironmentalPredicate.CRUISE_PHASE,
                        LogicalExpression(
                            operator=LogicalOperator.NOT,
                            predicates=[
                                LogicalExpression(
                                    operator=LogicalOperator.OR,
                                    predicates=[
                                        EnvironmentalPredicate.WIND_STRONG,
                                        EnvironmentalPredicate.OBSTACLES_IN_5M_RADIUS
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                output_actions=[Action.MAINTAIN],
                weight=4.7,
                description="Maintain efficient cruise altitude"
            )
        ])

        return rules

    def _initialize_mutual_exclusions(self) -> List[Set[Action]]:
        """Initialize sets of mutually exclusive actions."""
        return [
            {Action.ACCELERATE, Action.DECELERATE, Action.STOP, Action.EMERGENCY_STOP},
            {Action.UP, Action.DOWN},
            {Action.LEFT, Action.RIGHT},
            {Action.FORWARD, Action.BACKWARD},
            {Action.SLOW_DESCENT, Action.RAPID_DESCENT}
        ]
    
    def _is_valid_action_combination(self, actions: Set[Action]) -> bool:
        """Check if a combination of actions violates any mutual exclusion rules."""
        for exclusion_set in self.mutually_exclusive_actions:
            if len(actions.intersection(exclusion_set)) > 1:
                return False
        return True
    
    def _calculate_score(self, 
                        actions: Set[Action], 
                        active_predicates: Set[EnvironmentalPredicate],
                        planning_action: PlanningAction) -> float:
        """Calculate the score for a combination of actions given the current state."""
        score = 0.0
        
        for rule in self.rules:
            # Check if all conditions for this rule are met
            conditions_met = all(pred in active_predicates for pred in rule.conditions)
            
            if conditions_met:
                # Check if planning action matches (if rule has a planning action)
                planning_matches = (rule.planning_action is None or 
                                 rule.planning_action == planning_action)
                
                if planning_matches:
                    # Check if the rule's output actions are present in our action combination
                    if rule.output_actions is not None:
                        if all(action in actions for action in rule.output_actions):
                            score += rule.weight
                        else:
                            score -= rule.weight
        
        return score
    
    def verify_and_modify_action(self, 
                               planning_action: PlanningAction,
                               state_data: Dict) -> Set[Action]:
        """
        Verify the planned action and return the safest set of actions based on current state.
        
        Args:
            planning_action: The originally planned action
            state_data: Dictionary containing current state information
            
        Returns:
            Set of verified safe actions to execute
        """
        active_predicates = self.evaluate_environmental_predicates(state_data)
        
        # Generate all possible action combinations (up to 3 simultaneous actions)
        all_actions = list(Action)
        best_actions = set()
        best_score = float('-inf')
        
        # Try combinations of 1-3 actions
        for i in range(1, 4):
            for combo in self._generate_action_combinations(all_actions, i):
                action_set = set(combo)
                
                # Skip invalid combinations
                if not self._is_valid_action_combination(action_set):
                    continue
                
                # Calculate score for this combination
                score = self._calculate_score(action_set, active_predicates, planning_action)
                
                if score > best_score:
                    best_score = score
                    best_actions = action_set
        
        return best_actions
    
    def _generate_action_combinations(self, actions: List[Action], n: int) -> List[Tuple[Action, ...]]:
        """Generate all possible combinations of n actions."""
        if n == 1:
            return [(action,) for action in actions]
        
        combinations = []
        for i, action in enumerate(actions):
            for sub_combo in self._generate_action_combinations(actions[i+1:], n-1):
                combinations.append((action,) + sub_combo)
        
        return combinations

    def verify_safety(self, 
                     planning_action: PlanningAction,
                     state_data: Dict,
                     current_waypoint: np.ndarray,
                     next_waypoint: np.ndarray) -> SafetyVerificationResult:
        """Perform complete safety verification with input validation."""
        try:
            # Validate inputs
            if not isinstance(planning_action, PlanningAction):
                raise ValueError("planning_action must be a PlanningAction enum")
            
            if not isinstance(state_data, dict):
                raise ValueError("state_data must be a dictionary")
            
            if not isinstance(current_waypoint, np.ndarray) or current_waypoint.shape != (3,):
                raise ValueError("current_waypoint must be a numpy array of shape (3,)")
            
            if not isinstance(next_waypoint, np.ndarray) or next_waypoint.shape != (3,):
                raise ValueError("next_waypoint must be a numpy array of shape (3,)")

            # Get active predicates with error handling
            try:
                active_predicates = self.evaluate_environmental_predicates(state_data)
            except Exception as e:
                self.logger.error(f"Failed to evaluate predicates: {str(e)}")
                raise SafetyVerificationError(f"Predicate evaluation failed: {str(e)}")

            # Find best actions with error handling
            try:
                verified_actions, score, triggered_rules = self._find_best_actions(
                    planning_action, active_predicates
                )
            except Exception as e:
                self.logger.error(f"Failed to find best actions: {str(e)}")
                raise SafetyVerificationError(f"Action verification failed: {str(e)}")

            # Generate safety warnings
            safety_warnings = self._generate_safety_warnings(
                active_predicates, verified_actions, state_data
            )

            # Evaluate safety with error handling
            try:
                is_safe = self._evaluate_safety(
                    verified_actions, active_predicates, score, safety_warnings
                )
            except Exception as e:
                self.logger.error(f"Failed to evaluate safety: {str(e)}")
                raise SafetyVerificationError(f"Safety evaluation failed: {str(e)}")

            result = SafetyVerificationResult(
                verified_actions=verified_actions,
                score=score,
                active_predicates=active_predicates,
                triggered_rules=triggered_rules,
                is_safe=is_safe,
                safety_warnings=safety_warnings
            )

            # Log the verification result
            self.log_verification_result(result)

            return result

        except Exception as e:
            self.logger.error(f"Safety verification failed: {str(e)}")
            # Return a safe default result in case of error
            return self._get_safe_default_result()

    def _get_safe_default_result(self) -> SafetyVerificationResult:
        """Return a safe default result in case of errors."""
        return SafetyVerificationResult(
            verified_actions={Action.HOVER_IN_PLACE},
            score=0.0,
            active_predicates=set(),
            triggered_rules=[],
            is_safe=False,
            safety_warnings=["Safety verification failed, defaulting to hover in place"]
        )

    def convert_actions_to_waypoint_adjustment(self, 
                                         actions: Set[Action],
                                         current_waypoint: np.ndarray,
                                         next_waypoint: np.ndarray) -> np.ndarray:
        """
        Convert high-level safety actions to concrete waypoint adjustments.
        
        The conversion follows a priority-based system:
        1. Emergency actions (highest priority)
        2. Special movements (hover, descent)
        3. Movement scale modifications
        4. Directional adjustments
        5. Stability adjustments
        
        Args:
            actions: Set of high-level actions from safety verification
            current_waypoint: Current position as numpy array [x, y, z]
            next_waypoint: Target waypoint as numpy array [x, y, z]
            
        Returns:
            modified_waypoint: Adjusted waypoint that satisfies safety actions
        """
        try:
            # Input validation
            if not isinstance(actions, set):
                raise ValueError("actions must be a set")
            if not all(isinstance(a, Action) for a in actions):
                raise ValueError("all actions must be Action enum members")
            
            if not isinstance(current_waypoint, np.ndarray) or current_waypoint.shape != (3,):
                raise ValueError("current_waypoint must be a numpy array of shape (3,)")
            
            if not isinstance(next_waypoint, np.ndarray) or next_waypoint.shape != (3,):
                raise ValueError("next_waypoint must be a numpy array of shape (3,)")

            # Initialize modified waypoint and compute direction
            modified_waypoint = next_waypoint.copy()
            direction_vector = next_waypoint - current_waypoint
            distance = np.linalg.norm(direction_vector)
            unit_direction = direction_vector / (distance if distance > 0 else 1)

            # 1. Emergency Actions (Highest Priority)
            if Action.EMERGENCY_STOP in actions:
                return current_waypoint
            
            if Action.EMERGENCY_LANDING_REQUIRED in actions:
                emergency_descent_rate = 0.5  # m/s
                return np.array([
                    current_waypoint[0],
                    current_waypoint[1],
                    max(0, current_waypoint[2] - emergency_descent_rate)
                ])

            # 2. Special Movements
            if Action.HOVER_IN_PLACE in actions:
                hover_radius = 0.1  # Small movement radius for stability
                return current_waypoint + hover_radius * unit_direction

            if Action.STABILIZE_POSITION in actions:
                stabilize_radius = 0.2
                return current_waypoint + stabilize_radius * unit_direction

            # 3. Movement Scale Modifications
            movement_scale = 1.0
            if Action.DECELERATE in actions:
                movement_scale *= 0.5
            if Action.ACCELERATE in actions:
                movement_scale *= 1.5
            if Action.STOP in actions:
                movement_scale *= 0.1

            # Apply base movement scale
            modified_waypoint = current_waypoint + movement_scale * direction_vector

            # 4. Vertical Adjustments
            if Action.SLOW_DESCENT in actions:
                # Maintain horizontal position with slow descent
                descent_rate = 0.3
                modified_waypoint[0:2] = current_waypoint[0:2] + 0.3 * direction_vector[0:2]
                height_diff = next_waypoint[2] - current_waypoint[2]
                if height_diff < 0:
                    modified_waypoint[2] = current_waypoint[2] - min(descent_rate, abs(height_diff))

            if Action.RAPID_DESCENT in actions:
                # Faster descent while maintaining some horizontal movement
                rapid_descent_rate = 1.0
                modified_waypoint[0:2] = current_waypoint[0:2] + 0.2 * direction_vector[0:2]
                modified_waypoint[2] = current_waypoint[2] - rapid_descent_rate

            # 5. Directional Adjustments
            adjustment_magnitude = 0.5
            if Action.UP in actions:
                modified_waypoint[2] += adjustment_magnitude
            if Action.DOWN in actions:
                modified_waypoint[2] = max(0, modified_waypoint[2] - adjustment_magnitude)
            if Action.RIGHT in actions:
                modified_waypoint[1] += adjustment_magnitude
            if Action.LEFT in actions:
                modified_waypoint[1] -= adjustment_magnitude
            if Action.FORWARD in actions:
                modified_waypoint[0] += adjustment_magnitude
            if Action.BACKWARD in actions:
                modified_waypoint[0] -= adjustment_magnitude

            # 6. Safety Constraints
            # Ensure minimum movement for stability
            min_movement = 0.1
            if np.linalg.norm(modified_waypoint - current_waypoint) < min_movement:
                modified_waypoint = current_waypoint + min_movement * unit_direction

            # Ensure maximum movement for safety
            max_movement = 2.0
            movement_vector = modified_waypoint - current_waypoint
            movement_distance = np.linalg.norm(movement_vector)
            if movement_distance > max_movement:
                movement_direction = movement_vector / movement_distance
                modified_waypoint = current_waypoint + max_movement * movement_direction

            # Ensure minimum altitude
            min_altitude = 0.5
            modified_waypoint[2] = max(min_altitude, modified_waypoint[2])

            return modified_waypoint

        except Exception as e:
            self.logger.error(f"Failed to convert actions to waypoint: {str(e)}")
            # Return current waypoint as safe default
            return current_waypoint

    def _find_best_actions(self, 
                          planning_action: PlanningAction,
                          active_predicates: Set[EnvironmentalPredicate]) -> Tuple[Set[Action], float, List[Rule]]:
        """Find the best combination of actions and return details."""
        all_actions = list(Action)
        best_actions = set()
        best_score = float('-inf')
        triggered_rules = []
        
        # Try combinations of 1-3 actions
        for i in range(1, 4):
            for combo in self._generate_action_combinations(all_actions, i):
                action_set = set(combo)
                
                # Skip invalid combinations
                if not self._is_valid_action_combination(action_set):
                    continue
                
                # Calculate score and get triggered rules
                score, rules = self._calculate_score_with_rules(
                    action_set, active_predicates, planning_action
                )
                
                if score > best_score:
                    best_score = score
                    best_actions = action_set
                    triggered_rules = rules
        
        return best_actions, best_score, triggered_rules

    def _calculate_score_with_rules(self,
                                  actions: Set[Action],
                                  active_predicates: Set[EnvironmentalPredicate],
                                  planning_action: PlanningAction) -> Tuple[float, List[Rule]]:
        """Calculate score and return triggered rules."""
        score = 0.0
        triggered_rules = []
        
        for rule in self.rules:
            # Check if all conditions for this rule are met
            conditions_met = all(pred in active_predicates for pred in rule.conditions)
            
            if conditions_met:
                # Check if planning action matches
                planning_matches = (rule.planning_action is None or 
                                 rule.planning_action == planning_action)
                
                if planning_matches:
                    # Check if the rule's output actions are present
                    if rule.output_actions is not None:
                        if all(action in actions for action in rule.output_actions):
                            score += rule.weight
                            triggered_rules.append(rule)
                        else:
                            score -= rule.weight
        
        return score, triggered_rules

    def _generate_safety_warnings(self,
                                active_predicates: Set[EnvironmentalPredicate],
                                verified_actions: Set[Action],
                                state_data: Dict) -> List[str]:
        """Generate safety warnings based on current state and actions."""
        warnings = []
        
        # Critical safety warnings
        if EnvironmentalPredicate.OBSTACLES_IN_1M_RADIUS in active_predicates:
            warnings.append("CRITICAL: Obstacle detected in immediate vicinity")
            
        if EnvironmentalPredicate.BATTERY_CRITICAL in active_predicates:
            warnings.append("CRITICAL: Battery level critical")
            
        if EnvironmentalPredicate.GPS_SIGNAL_WEAK in active_predicates:
            warnings.append("WARNING: GPS signal weak, position accuracy reduced")
        
        # Landing-related warnings
        if (EnvironmentalPredicate.LANDING in active_predicates and 
            EnvironmentalPredicate.WIND_STRONG in active_predicates):
            warnings.append("CAUTION: Strong winds during landing phase")
            
        if (EnvironmentalPredicate.LANDING in active_predicates and 
            not EnvironmentalPredicate.LANDING_ZONE_CLEAR in active_predicates):
            warnings.append("WARNING: Landing zone not confirmed clear")
        
        # Stability warnings
        if (EnvironmentalPredicate.YAW_RATE_HIGH in active_predicates or
            EnvironmentalPredicate.ROLL_RATE_HIGH in active_predicates or
            EnvironmentalPredicate.PITCH_RATE_HIGH in active_predicates):
            warnings.append("CAUTION: Vehicle attitude rates high")
            
        # Environmental warnings
        if EnvironmentalPredicate.VISIBILITY_LOW in active_predicates:
            warnings.append("CAUTION: Low visibility conditions")
            
        if EnvironmentalPredicate.WIND_GUST_DETECTED in active_predicates:
            warnings.append("CAUTION: Wind gusts detected")
        
        return warnings

    def _evaluate_safety(self,
                        verified_actions: Set[Action],
                        active_predicates: Set[EnvironmentalPredicate],
                        score: float,
                        warnings: List[str]) -> bool:
        """Evaluate if the current state and actions are safe."""
        # Critical safety violations
        if (EnvironmentalPredicate.OBSTACLES_IN_1M_RADIUS in active_predicates and 
            Action.EMERGENCY_STOP not in verified_actions):
            return False
            
        if (EnvironmentalPredicate.BATTERY_CRITICAL in active_predicates and 
            Action.EMERGENCY_LANDING_REQUIRED not in verified_actions):
            return False
            
        # Check for multiple high-risk conditions
        high_risk_predicates = {
            EnvironmentalPredicate.GPS_SIGNAL_WEAK,
            EnvironmentalPredicate.VISIBILITY_LOW,
            EnvironmentalPredicate.WIND_STRONG,
            EnvironmentalPredicate.YAW_RATE_HIGH,
            EnvironmentalPredicate.ROLL_RATE_HIGH,
            EnvironmentalPredicate.PITCH_RATE_HIGH
        }
        
        risk_count = len(active_predicates.intersection(high_risk_predicates))
        if risk_count >= 3:  # If 3 or more high-risk conditions are active
            return False
            
        # Check if score is too low
        if score < -5.0:  # Threshold for accumulated negative scores
            return False
            
        # Check for too many warnings
        if len(warnings) >= 5:  # If too many active warnings
            return False
            
        return True

    def log_verification_result(self, result: SafetyVerificationResult):
        """Log the results of safety verification."""
        self.logger.info("Safety Verification Results:")
        self.logger.info(f"Is Safe: {result.is_safe}")
        self.logger.info(f"Score: {result.score}")
        
        if result.safety_warnings:
            self.logger.warning("Safety Warnings:")
            for warning in result.safety_warnings:
                self.logger.warning(f"  - {warning}")
                
        self.logger.info("Active Predicates:")
        for predicate in result.active_predicates:
            self.logger.info(f"  - {predicate.name}")
            
        self.logger.info("Verified Actions:")
        for action in result.verified_actions:
            self.logger.info(f"  - {action.name}")
            
        self.logger.info("Triggered Rules:")
        for rule in result.triggered_rules:
            conditions = [c.name for c in rule.conditions]
            actions = [a.name for a in rule.output_actions] if rule.output_actions else []
            self.logger.info(f"  - Rule(weight={rule.weight}):")
            self.logger.info(f"    Conditions: {conditions}")
            self.logger.info(f"    Actions: {actions}")

    def evaluate_logical_expression(self, 
                                  expr: LogicalExpression, 
                                  state_data: Dict,
                                  active_predicates: Set[EnvironmentalPredicate]) -> bool:
        """Evaluates complex logical expressions"""
        
        if expr.operator == LogicalOperator.AND:
            return all(self.evaluate_predicate(p, state_data, active_predicates) 
                      for p in expr.predicates)
        
        elif expr.operator == LogicalOperator.OR:
            return any(self.evaluate_predicate(p, state_data, active_predicates) 
                      for p in expr.predicates)
        
        elif expr.operator == LogicalOperator.NOT:
            return not self.evaluate_predicate(expr.predicates[0], state_data, active_predicates)
        
        elif expr.operator == LogicalOperator.IMPLIES:
            antecedent = self.evaluate_predicate(expr.predicates[0], state_data, active_predicates)
            consequent = self.evaluate_predicate(expr.predicates[1], state_data, active_predicates)
            return (not antecedent) or consequent
        
        elif expr.operator == LogicalOperator.XOR:
            results = [self.evaluate_predicate(p, state_data, active_predicates) 
                      for p in expr.predicates]
            return sum(results) == 1
        
        elif expr.operator == LogicalOperator.FORALL:
            domain = self.get_domain(expr.variables[0], state_data)
            return all(self.evaluate_predicate_with_binding(expr.predicates[0], 
                                                          expr.variables[0], 
                                                          item,
                                                          state_data,
                                                          active_predicates)
                      for item in domain)
        
        elif expr.operator == LogicalOperator.EXISTS:
            domain = self.get_domain(expr.variables[0], state_data)
            return any(self.evaluate_predicate_with_binding(expr.predicates[0], 
                                                          expr.variables[0], 
                                                          item,
                                                          state_data,
                                                          active_predicates)
                      for item in domain)
        
        elif expr.operator == LogicalOperator.NAND:
            return not all(self.evaluate_predicate(p, state_data, active_predicates) 
                         for p in expr.predicates)
        
        elif expr.operator == LogicalOperator.NOR:
            return not any(self.evaluate_predicate(p, state_data, active_predicates) 
                         for p in expr.predicates)

    def get_domain(self, variable: Variable, state_data: Dict) -> List:
        """Returns the domain for a variable based on its type"""
        if variable.domain == 'Obstacles':
            return state_data.get('obstacles', [])
        elif variable.domain == 'Waypoints':
            return state_data.get('waypoints', [])
        return [] 