import streamlit as st
from crewai import Crew
from textwrap import dedent
from agents import TravelAgents
from tasks import TravelTasks
from dotenv import load_dotenv
import os

# Load environment variables and set up credentials
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service.json"

# Define the TripCrew class
class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.origin = origin
        self.cities = cities
        self.date_range = date_range
        self.interests = interests

    def run(self):
        agents = TravelAgents()
        tasks = TravelTasks()

        expert_travel_agent = agents.expert_travel_agent()
        city_selection_expert = agents.city_selection_expert()
        local_tour_guide = agents.local_tour_guide()

        plan_itinerary = tasks.plan_itinerary(
            expert_travel_agent,
            self.cities,
            self.date_range,
            self.interests
        )

        identify_city = tasks.identify_city(
            city_selection_expert,
            self.origin,
            self.cities,
            self.interests,
            self.date_range
        )

        gather_city_info = tasks.gather_city_info(
            local_tour_guide,
            self.cities,
            self.date_range,
            self.interests
        )

        crew = Crew(
            agents=[expert_travel_agent, city_selection_expert, local_tour_guide],
            tasks=[plan_itinerary, identify_city, gather_city_info],
            verbose=True,
        )

        result = crew.kickoff()
        return result

# Streamlit app UI
st.title("Trip Planner Crew")

origin = st.text_input("From where will you be traveling from?")
cities = st.text_input("What are the cities options you are interested in visiting?")
date_range = st.text_input("What is the date range you are interested in traveling?")
interests = st.text_input("What are some of your high level interests and hobbies?")

if st.button("Plan Trip"):
    with st.spinner("Planning your trip..."):
        trip_crew = TripCrew(origin, cities, date_range, interests)
        result = trip_crew.run()
    st.markdown("## Here is your Trip Plan")
    st.write(result)
