import googlemaps
import requests
import geopy.distance
import time

API_KEY = "AIzaSyB-CDOUh12fNQbLCWIA4eoY2xms8r1Yags"
gmaps = googlemaps.Client(key=API_KEY)

origin = "37.7749,-122.4194"  # Example: San Francisco (latitude, longitude)
destination = "37.8715,-122.2730"  # Example: Berkeley

def get_directions(origin, destination): #return nothing, just prints directionsl line by line
    directions = gmaps.directions(origin, destination, mode="driving")
    for step in directions[0]['legs'][0]['steps']:
        print(step['html_instructions'], step['distance']['text'])
    
def get_gps_location():
    #gets the gps location, 

    send_time = time.time()  
    print(f"Request sent at: {send_time:.6f} seconds")
    url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={API_KEY}" # link to geolocation API
    response = requests.post(url, json={"considerIP": True})
    receive_time = time.time()  
    print(f"Response received at: {receive_time:.6f} seconds")
    delay = receive_time - send_time
    print(f"API Response Delay: {delay:.6f} seconds")
    return response.json()["location"]

def get_distance_to_turn(current_loc, turn_loc):
    return geopy.distance.geodesic((current_loc["lat"], current_loc["lng"]), (turn_loc["lat"], turn_loc["lng"])).feet


current_location = get_gps_location()
turn_location = {"lat": 37.8715, "lng": -122.2730}  # Example: Berkeley
distance_to_turn = get_distance_to_turn(current_location, turn_location)
print (f"Distance to turn: {distance_to_turn} feet")