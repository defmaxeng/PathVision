import os
import json
import openrouteservice
import folium
import json

# API_KEY = os.getenv("OPENROUTESERVICE_API_KEY")
# if not API_KEY:
#     raise RuntimeError("Set OPENROUTESERVICE_API_KEY first.")

# Example: Puerta del Sol → Museo del Prado (Madrid)
start = (-74.35230, 40.72061)   # lon, lat
end   = (-74.39890, 40.65127)

client = openrouteservice.Client(key="eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImE1MjM3NzQxNmFhNzQ4Y2E5Y2MwY2U0NTcwOWZjNmExIiwiaCI6Im11cm11cjY0In0=")

# Request a driving route and get GeoJSON back
route_geojson = client.directions(
    coordinates=[start, end],
    profile="driving-car",
    format="geojson",
    instructions=True
)

# Pull summary info
feat = route_geojson["features"][0]
summary = feat["properties"]["summary"]
distance_km = summary["distance"] / 1000
duration_min = summary["duration"] / 60

print(f"Distance: {distance_km:.2f} km")
print(f"Duration: {duration_min:.1f} min")

# Center map near the start point
m = folium.Map(location=[start[1], start[0]], zoom_start=14)

# Add route line
folium.GeoJson(route_geojson, name="route").add_to(m)
with open("route.geojson", "w") as f:
    json.dump(route_geojson, f, indent=2)
# Add markers
folium.Marker([start[1], start[0]], tooltip="Start: Puerta del Sol").add_to(m)
folium.Marker([end[1], end[0]], tooltip="End: Museo del Prado").add_to(m)

# Turn-by-turn steps (optional): add as a popup list
steps = []
for seg in feat["properties"]["segments"]:
    for st in seg["steps"]:
        steps.append(st["instruction"])

print(steps)

html_steps = "<br>".join(steps)
folium.Marker(
    [ (start[1]+end[1])/2, (start[0]+end[0])/2 ],
    popup=folium.Popup(f"<b>Instructions</b><br>{html_steps}", max_width=450)
).add_to(m)

m.save("route_map.html")
print("Saved map → route_map.html")
