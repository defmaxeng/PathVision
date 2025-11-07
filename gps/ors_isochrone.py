import os
import openrouteservice
import folium

# API_KEY = os.getenv("OPENROUTESERVICE_API_KEY")
# if not API_KEY:
#     raise RuntimeError("Set OPENROUTESERVICE_API_KEY first.")

client = openrouteservice.Client(key="eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImE1MjM3NzQxNmFhNzQ4Y2E5Y2MwY2U0NTcwOWZjNmExIiwiaCI6Im11cm11cjY0In0=")

# Plaza Mayor, Madrid
center = (-3.707398, 40.415363)  # lon, lat

iso = client.isochrones(
    locations=[center],
    profile="foot-walking",
    range=[600],  # seconds → 10 minutes
    range_type="time",
    attributes=["area","reachfactor"]
)

m = folium.Map(location=[center[1], center[0]], zoom_start=15)
folium.GeoJson(iso, name="10-min walk area").add_to(m)
folium.Marker([center[1], center[0]], tooltip="Center").add_to(m)

m.save("isochrone_10min.html")
print("Saved map → isochrone_10min.html")
