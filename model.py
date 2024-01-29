# Importing necessary libraries
import networkx as nx
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from statistics import mean
import geopandas as gpd
import rasterio as rs
import matplotlib.pyplot as plt
import random

# Import the agent class(es) from agents.py
from agents import Households

# Import functions from functions.py
from functions import get_flood_map_data, calculate_basic_flood_damage
from functions import map_domain_gdf, floodplain_gdf


# Define the AdaptationModel class
class AdaptationModel(Model):
    """
    The main model running the simulation. It sets up the network of household agents,
    simulates their behavior, and collects data. The network type can be adjusted based on study requirements.
    """

    def __init__(self, 
                 seed = None,
                 number_of_households = 50, # number of household agents
                 # Simplified argument for choosing flood map. Can currently be "harvey", "100yr", or "500yr"
                 flood_map_choice='harvey',
                 # Step at which the flood happens
                 step_of_flood=5,

                 ### Network related parameters ###
                 # The social network structure that is used.
                 # Can currently be "erdos_renyi", "barabasi_albert", "watts_strogatz", or "no_network"
                 network = 'watts_strogatz',
                 # Likeliness of edge being created between two nodes
                 probability_of_network_connection = 0.4,
                 # Number of edges for BA network
                 number_of_edges = 3,
                 # Number of nearest neighbours for WS social network
                 number_of_nearest_neighbours = 5,

                 ### Agent related parameters ###
                 # Households have an initial saving based on their house size (small, medium, large)
                 initial_savings_small = 1000,
                 initial_savings_medium = 2500,
                 initial_savings_large = 5000,
                 # Amount of economic damage in the worst case. Used for calculating economic damage
                 economic_damage_multiplier = 340000,
                 # Cost of taking a measure that decreases flood depth estimated by 1 cm
                 measure_costs = 10000,
                 # Government gives a subsidy every quarter of a year (every tick)
                 # $1500 dollar per tick, so $6000 per year
                 government_subsidy = 1500,
                 # Threshold for an agent to be influenced by a neighbour
                 influential_threshold = 0.2,
                 # When an agent is influenced by its neighbour,
                 # it adds or subtracts a percentage of the neighbours likelihood to its own likelihood
                 percentage_of_influence = 0.1, # this is equal to 10%
                 # A multiplier to indicate the minimum amount of savings needed to adapt
                 savings_threshold = 2, # at least 2 times the cost of a measure are needed
                 # Percentage of savings that an agent spends on measures in one step
                 percentage_savings_spent = 0.5 # this is equal to 50%
                 ):
        
        super().__init__(seed = seed)
        
        # Defining the variables and setting the values
        self.seed = seed
        self.number_of_households = number_of_households
        self.step_of_flood = step_of_flood
        self.initial_savings_small = initial_savings_small
        self.initial_savings_medium = initial_savings_medium
        self.initial_savings_large = initial_savings_large
        self.economic_damage_multiplier = economic_damage_multiplier
        self.measure_costs = measure_costs
        self.government_subsidy = government_subsidy
        self.influential_threshold = influential_threshold
        self.percentage_of_influence = percentage_of_influence
        self.savings_threshold = savings_threshold
        self.percentage_savings_spent = percentage_savings_spent
        self.running = True

        # Network
        self.network = network
        self.probability_of_network_connection = probability_of_network_connection
        self.number_of_edges = number_of_edges
        self.number_of_nearest_neighbours = number_of_nearest_neighbours

        # Generating the graph according to the network used and the network parameters specified
        self.G = self.initialize_network()
        # Create grid out of network graph
        self.grid = NetworkGrid(self.G)

        # Initialize maps
        self.initialize_maps(flood_map_choice)

        # Set schedule for agents
        self.schedule = RandomActivation(self)  # Schedule for activating agents

        # Create households through initiating a household on each node of the network graph
        for i, node in enumerate(self.G.nodes()):
            household = Households(unique_id=i, model=self)
            self.schedule.add(household)
            self.grid.place_agent(agent=household, node_id=node)

        # Data collection setup to collect data
        model_metrics = {
                        "total_adapted_households": self.total_adapted_households,
                        "total_expected_value_of_economic_damage" : self.total_expected_value_of_economic_damage,
                        "average_flood_damage_estimated": self.calculate_average_flood_damage_estimated
                        }
        
        agent_metrics = {
                        "FloodDepthEstimated": "flood_depth_estimated",
                        "FloodDamageEstimated" : "flood_damage_estimated",
                        "FloodDepthActual": "flood_depth_actual",
                        "FloodDamageActual" : "flood_damage_actual",
                        "IsAdapted": "is_adapted",
                        "location":"location",
                        "Savings": "savings",
                        "LikelihoodToAdapt": 'likelihood_adapt'}

        #Set up the data collector
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)

    def initialize_network(self):
        """
        Initialize and return the social network graph based on the provided network type using pattern matching.
        """
        if self.network == 'erdos_renyi':
            return nx.erdos_renyi_graph(n=self.number_of_households,
                                        p=self.number_of_nearest_neighbours / self.number_of_households,
                                        seed=self.seed)
        elif self.network == 'barabasi_albert':
            return nx.barabasi_albert_graph(n=self.number_of_households,
                                            m=self.number_of_edges,
                                            seed=self.seed)
        elif self.network == 'watts_strogatz':
            return nx.watts_strogatz_graph(n=self.number_of_households,
                                        k=self.number_of_nearest_neighbours,
                                        p=self.probability_of_network_connection,
                                        seed=self.seed)
        elif self.network == 'no_network':
            G = nx.Graph()
            G.add_nodes_from(range(self.number_of_households))
            return G
        else:
            raise ValueError(f"Unknown network type: '{self.network}'. "
                            f"Currently implemented network types are: "
                            f"'erdos_renyi', 'barabasi_albert', 'watts_strogatz', and 'no_network'")

    def initialize_maps(self, flood_map_choice):
        """
        Initialize and set up the flood map related data based on the provided flood map choice.
        """
        # Define paths to flood maps
        flood_map_paths = {
            'harvey': r'../input_data/floodmaps/Harvey_depth_meters.tif',
            '100yr': r'../input_data/floodmaps/100yr_storm_depth_meters.tif',
            '500yr': r'../input_data/floodmaps/500yr_storm_depth_meters.tif'  # Example path for 500yr flood map
        }

        # Throw a ValueError if the flood map choice is not in the dictionary
        if flood_map_choice not in flood_map_paths.keys():
            raise ValueError(f"Unknown flood map choice: '{flood_map_choice}'. "
                             f"Currently implemented choices are: {list(flood_map_paths.keys())}")

        # Choose the appropriate flood map based on the input choice
        flood_map_path = flood_map_paths[flood_map_choice]

        # Loading and setting up the flood map
        self.flood_map = rs.open(flood_map_path)
        self.band_flood_img, self.bound_left, self.bound_right, self.bound_top, self.bound_bottom = get_flood_map_data(
            self.flood_map)

    def total_adapted_households(self):
        """Return the total number of households that have adapted"""
        adapted_count = sum([1 for agent in self.schedule.agents if isinstance(agent, Households) and agent.is_adapted])
        return adapted_count

    def total_expected_value_of_economic_damage(self):
        """Return the total expected economic damage"""
        expected_economic_damage = sum([agent.calculate_economic_damage() for agent in self.schedule.agents if isinstance(agent, Households)])
        return expected_economic_damage

    def calculate_average_flood_damage_estimated(self):
        """Return the average flood damage estimated"""
        average_flood_damage_estimated = mean([agent.flood_damage_estimated for agent in self.schedule.agents if isinstance(agent, Households)])
        return average_flood_damage_estimated

    def plot_model_domain_with_agents(self):
        fig, ax = plt.subplots()
        # Plot the model domain
        map_domain_gdf.plot(ax=ax, color='lightgrey')
        # Plot the floodplain
        floodplain_gdf.plot(ax=ax, color='lightblue', edgecolor='k', alpha=0.5)

        # Collect agent locations and statuses
        for agent in self.schedule.agents:
            color = 'blue' if agent.is_adapted else 'red'
            ax.scatter(agent.location.x, agent.location.y, color=color, s=10, label=color.capitalize() if not ax.collections else "")
            ax.annotate(str(agent.unique_id), (agent.location.x, agent.location.y), textcoords="offset points", xytext=(0,1), ha='center', fontsize=9)
        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Red: not adapted, Blue: adapted")

        # Customize plot with titles and labels
        plt.title(f'Model Domain with Agents at Step {self.schedule.steps}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def step(self):
        """
        Introducing a shock:
        At time step 5, there will be a global flooding.
        This will result in actual flood depth. Here, we assume it is a random number
        between 0.5 and 1.2 of the estimated flood depth.
        This actual flood dpeth will also result in an actual flood damage.
        """
        if self.schedule.steps == self.step_of_flood:
            for agent in self.schedule.agents:
                # Calculate the actual flood depth as a random number between 0.5 and 1.2 times the estimated flood depth
                agent.flood_depth_actual = random.uniform(0.5, 1.2) * agent.flood_depth_estimated
                # calculate the actual flood damage given the actual flood depth
                agent.flood_damage_actual = calculate_basic_flood_damage(agent.flood_depth_actual)
        
        # Collect data and advance the model by one step
        self.datacollector.collect(self)
        self.schedule.step()
