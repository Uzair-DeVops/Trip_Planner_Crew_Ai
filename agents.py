import os
from crewai import Agent
from textwrap import dedent
from tools.search_tools import SearchTools
from tools.calculator_tools import CalculatorTools
from crewai import LLM



class TravelAgents:
    def __init__(self):
        google_api_key = os.getenv("GEMINI_API_KEY")  # Ensure API key is set in the environment

        self.OpenAIGPT35 = LLM(
            model="gemini/gemini-2.0-flash-exp", temperature=0.7, api_key="AIzaSyDlGuiJOqQePVsQEu5gWiftb74RDGvcq-c",

        )
        self.OpenAIGPT4 = LLM(
            model="gemini/gemini-1.5-flash", temperature=0.7, api_key="AIzaSyDlGuiJOqQePVsQEu5gWiftb74RDGvcq-c",

        )

        # Instantiate tools
        self.search_tool = SearchTools()
        self.calculator_tool = CalculatorTools()

    def expert_travel_agent(self):
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent("""
                Expert in travel planning and logistics. 
                I have decades of experience making travel itineraries.
            """),
            goal=dedent("""
                Create a 7-day travel itinerary with detailed per-day plans,
                including budget, packing suggestions, and safety tips.
            """),
            tools=[self.search_tool.search_internet, self.calculator_tool.calculate],
            verbose=True,
            llm=self.OpenAIGPT4,
        )

    def city_selection_expert(self):
        return Agent(
            role="City Selection Expert",
            backstory=dedent("""
                Expert at analyzing travel data to pick ideal destinations.
            """),
            goal=dedent("""
                Select the best cities based on weather, season, prices, and traveler interests.
            """),
            tools=[self.search_tool.search_internet],
            verbose=True,
            llm=self.OpenAIGPT4,
        )

    def local_tour_guide(self):
        return Agent(
            role="Local Tour Guide",
            backstory=dedent("""
                Knowledgeable local guide with extensive information
                about the city, its attractions, and customs.
            """),
            goal=dedent("""
                Provide the BEST insights about the selected city.
            """),
            tools=[self.search_tool.search_internet],
            verbose=True,
            llm=self.OpenAIGPT4,
        )
