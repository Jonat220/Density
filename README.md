# Building Density Calculator

A Streamlit app that counts OpenStreetMap buildings within a radius, computes density per area, and analyzes building heights, with an interactive map.

## Run
```powershell
cd D:\Density
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Use
- In the sidebar: choose Address or Coordinates
- Enter address or lat/lon
- Set radius and units (km/mi)
- Click "Calculate"

If nothing appears, ensure you clicked Calculate and check the terminal for errors. Overpass API limits can cause temporary failures. Try a smaller radius first.

## Features
- **Building Analysis**: 
  - Building count and density calculations (per km² and mi²)
  - Roof area analysis with total and average building areas
  - Total cumulative building height calculations
- **Enhanced Height Analysis**: 
  - Multiple data sources (OSM tags, building levels, satellite elevation)
  - Confidence scoring for data quality
  - Data source breakdown and recommendations
- **Road & Infrastructure Analysis**:
  - Road length and area calculations (tiled/paved vs untiled/unpaved)
  - Intersection counting with surface type classification
  - Footpath and walkway analysis
- **People & Phone Tracking**:
  - 15-minute interval tracking simulation
  - Real-time people and device counting
  - Historical data analysis with hourly averages
- **Interactive Map**: Visualizes the search area and building locations
- **Satellite Integration**: Optional NASA SRTM elevation data for enhanced accuracy

**Note**: Height data availability depends on OpenStreetMap contributors and satellite data coverage. The people/phone tracking is currently simulated for demonstration - in production it would integrate with real sensors, cameras, or network analytics. 
