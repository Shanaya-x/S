# Importing necessary libraries
import random
import numpy as np
from mesa import Agent
from shapely import contains_xy
from shapely.geometry import Point
from tabulate import tabulate

# Import functions from functions.py
from functions import generate_random_location_within_map_domain, get_flood_depth, calculate_basic_flood_damage, \
    floodplain_multipolygon


# Define the Households agent class
class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # Initial adaptation status set to False
        self.is_adapted = False

        # Getting flood map values
        # Get a random location on the map
        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)

        # Check whether the location is within floodplain
        self.in_floodplain = False
        if contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y):
            self.in_floodplain = True

        # Get the estimated flood depth at those coordinates. 
        # The estimated flood depth is calculated based on the flood map (i.e., past data) so this is not the actual flood depth
        # Flood depth can be negative if the location is at a high elevation
        self.flood_depth_estimated = get_flood_depth(corresponding_map=model.flood_map, location=self.location, band=model.band_flood_img)
        # handle negative values of flood depth
        if self.flood_depth_estimated < 0:
            self.flood_depth_estimated = 0

        # Calculate the estimated flood damage given the estimated flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated)

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since there is not flood yet,
        # and it will update its value when there is a shock (i.e., actual flood). Shock happens at some point during the simulation
        self.flood_depth_actual = 0

        # Calculate the actual flood damage given the actual flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_actual = calculate_basic_flood_damage(flood_depth=self.flood_depth_actual)

        # Calculate the initial economic damage. This is the estimated monetary damage to a household when a flood happens
        self.economic_damage = self.calculate_economic_damage()

        # A house size is randomly assigned. 1 = small, 2 = medium, 3 = large
        self.house_size = random.randint(1,3)

        # Calculate the initial savings based on the house size
        self.savings = self.calculate_initial_savings()

        # The attribute likelihood to adapt is equal to the flood_damage_estimated
        # During the model run the agent will be influenced by other agents, and the likelihood will change
        self.likelihood_adapt = self.flood_damage_estimated

        # This attributes keeps track of the neighbours of a household.
        # The neighbours are determined at step = 0, and stay the same for the entire model run
        self.neighbours=[]

        # This attribute is used to store the likelihood to adapt of the neighbour that is selected
        # At the beginning of the model, the agent has no contact with its neighbours, hence
        # the agent doesn't know the likelihood to adapt from its selected neighbour.
        self.likelihood_adapt_neighbour = 0

        # Keeps track of an agents is going to adapt or not in the current step
        self.ready_to_adapt = False

        table = [['Household', 'Likelihood to adapt neighbour', 'Economic damage', 'Ready to adapt', 'Is adapted'],
                 [{self.unique_id}, {self.likelihood_adapt_neighbour}, {self.economic_damage}, {self.ready_to_adapt}, {self.is_adapted}]]
        print(tabulate(table))

    def calculate_economic_damage(self):
        """Calculate the economic damage based on the flood damage estimated"""
        estimated_economic_damage = self.flood_damage_estimated * self.model.economic_damage_multiplier
        mu, sigma = estimated_economic_damage, 0.05
        probability_distribution = np.random.normal(mu, sigma, 1000)
        expected_value_of_economic_damage = random.choice(probability_distribution)
        if expected_value_of_economic_damage <0:
            expected_value_of_economic_damage = 0
        return expected_value_of_economic_damage

    def calculate_initial_savings(self):
        """The initial saving of a household is assigned based on the house size"""
        if self.house_size == 1:
            return self.model.initial_savings_small  #$1000
        elif self.house_size == 2:
            return self.model.initial_savings_medium #$2500
        elif self.house_size == 3:
            return self.model.initial_savings_large  #$5000

    def add_neighbours(self):
        """The neighbours of the agent are determined and saved as a list"""
        self.neighbours=self.model.grid.get_neighbors(self.pos, include_center=False)

    def save_money(self):
        """An agent receives government subsidy which the agent saves and adds to its current savings"""
        self.savings = self.savings + self.model.government_subsidy

    def pick_neighbour(self):
        """Randomly select one of the neighbouring agents"""
        if self.neighbours:
            chosen_neighbour = self.random.choice(self.neighbours)
            return chosen_neighbour
        else:
            return None

    def get_likelihood_neighbour(self):
        """Store the likelihood to adapt of the randomly picked neighbour"""
        self.likelihood_adapt_neighbour = self.pick_neighbour().likelihood_adapt

    def calculate_likelihood_adapt(self):
        """
        If the likelihood of the picked neighbour differs with the influential threshold (0.2 in the base case) from the
        own likelihood to adapt, the household will be influenced by the neighbour.
        Then, the likelihood of the agent will increase or decrease with a percentage of the likelihood to adapt of the neighbour
        (0.1 in the base case).
        """
        # If the neighbour is more likely to adapt, the likelihood of the agent self will increase
        if self.likelihood_adapt_neighbour - self.likelihood_adapt >= self.model.influential_threshold:
            self.likelihood_adapt += self.model.percentage_of_influence*self.likelihood_adapt_neighbour
        # If the neighbour is less likely to adapt, the likelihood of the agent self will decrease
        elif self.likelihood_adapt_neighbour - self.likelihood_adapt <= -self.model.influential_threshold:
            self.likelihood_adapt -= self.model.percentage_of_influence*self.likelihood_adapt_neighbour
        # If the likelihood does not differ more than the threshold, the likelihood will stay the same
        else:
            self.likelihood_adapt = self.likelihood_adapt

    def reconsider_adaptation_decision(self):
        """Based on its likelihood to adapt and its savings the agent decides if it wants to adapt"""
        # Only reconsider if the agent still has a flood depth estimated greater than 0
        if self.flood_depth_estimated > 0:
            # Check if the agent has enough money to adapt
            if self.savings >= self.model.savings_threshold* self.model.measure_costs:
                # Randomly draw a number between 0 and 1 to determine if the agent will adapt
                # A larger likelihood to adapt will increase the chance of adapting
                if self.likelihood_adapt >= np.random.rand():
                    self.ready_to_adapt = True
            else:
                self.ready_to_adapt = False

    def take_adaptation_measures(self):
        """If an agent is ready to adapt it will take measures, resulting in a smaller flood depth estimated"""
        if self.ready_to_adapt:
            # Determine how many centimeters the estimated flood depth will decrease rounded down to the nearest whole number
            adaptation_in_cm = self.savings*self.model.percentage_savings_spent // self.model.measure_costs

            # Taking adaptation measures costs money, so the savings decrease by the number of adapted centimeters * cost of 1 centimeter
            self.savings = self.savings - adaptation_in_cm * self.model.measure_costs

            # Decrease the flood depth estimated
            self.flood_depth_estimated = self.flood_depth_estimated - (adaptation_in_cm/100) # Change the adaptation in cm to meters

            # The flood depth estimated can't be negative
            if self.flood_depth_estimated < 0:
                self.flood_depth_estimated = 0

            # Update the flood damage estimated, based on the new flood depth estimated
            self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated)

            # Tracks if an agent has ever adapted
            self.is_adapted = True

            # Update the likelihood to adapt to the new flood_depth_estimated based on the flood_damage_estimated
            self.likelihood_adapt = self.flood_damage_estimated

            # After taking adaptation measures, the 'ready_to_adapt' is set back to False
            self.ready_to_adapt = False

    def step(self):
        """
        In a step the agent saves the subsidy it receives from the government,
        it has contact with its neighbour and might be influenced by their opinion.
        Then the agent reconsiders if they want to adapt, and if they decide to do so, they take adaptation measures.
        Finally, the economic damage is determined
        """

        # At the beginning of the model run the agent creates a list of all its neighbours
        if self.model.schedule.steps == 0:
            self.add_neighbours()

        # Agent saves subsidy from the government
        self.save_money()

        # Agent gets the likelihood to adapt of one of its neighbours
        self.get_likelihood_neighbour()

        # Agent determines its likelihood to adapt
        self.calculate_likelihood_adapt()

        # Agent determines if it wants to take adaptation measures in this step
        self.reconsider_adaptation_decision()

        # If agent decided to adapt, it will take adaptation measures
        self.take_adaptation_measures()

        # Calculate the economic damage of the agent
        self.economic_damage = self.calculate_economic_damage()
