from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import os, time, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium

from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "traffic_accident.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
MODEL_FILE = ARTIFACTS / "traffic_accident_model.joblib"
ENCODER_FILE = ARTIFACTS / "weather_encoder.joblib"
METRICS_FILE = ARTIFACTS / "model_metrics.joblib"
AP_CENTER = (15.9129, 79.7400)
AP_BBOX = (12.5, 17.2, 76.7, 84.8)  # south, north, west, east
FEATURES = ["Hour","Day_Number","Month","Year","Is_Weekend","Start_Lat","Start_Lng","Weather_Encoded"]
REQUIRED = ["Severity","Start_Time","Start_Lat","Start_Lng","Weather_Condition"]

st.set_page_config(page_title="Smart Traffic AP | Live Intelligence", page_icon="🚦", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#0a1020;color:#edf3ff}
[data-testid="stSidebar"]{background:#101a2e;border-right:1px solid #263653}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;color:#f4f7ff}
.hero{padding:24px 28px;border:1px solid #2a3b5d;border-radius:18px;background:linear-gradient(115deg,#17294b,#1b2441 60%,#202f50);margin-bottom:20px}
.kicker{color:#6fe2d2;font-size:12px;letter-spacing:2px;font-weight:700}
.hero h1{font-size:32px;margin:6px 0;color:#fff}
.sub{color:#b8c8e2}
.metric{background:#14213a;border:1px solid #2a3b5d;border-radius:14px;padding:16px}
.mlabel{font-size:11px;letter-spacing:1px;color:#9fb2d0;text-transform:uppercase}
.mvalue{font-family:'Space Grotesk';font-size:26px;color:white;font-weight:700}
.mnote{font-size:11px;color:#72dcca}
.smallnote{color:#9aacc8;font-size:12px}
div[data-testid="stForm"]{background:#131f36;border:1px solid #2a3b5d;padding:18px;border-radius:15px}
</style>
""", unsafe_allow_html=True)

def metric(label, value, note=""):
    st.markdown(f'<div class="metric"><div class="mlabel">{label}</div><div class="mvalue">{value}</div><div class="mnote">{note}</div></div>', unsafe_allow_html=True)

def http_get(url, params=None, timeout=18):
    try:
        r=requests.get(url,params=params,timeout=timeout,headers={"User-Agent":"SmartTrafficAPAcademicProject/1.0"})
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        message = str(e)
        secret = (params or {}).get("key")
        if secret:
            message = message.replace(str(secret), "[REDACTED]")
        return None, message

@st.cache_data(ttl=300, show_spinner=False)
def get_weather(lat, lon):
    # Open-Meteo current conditions; free API, no key for non-commercial use.
    url="https://api.open-meteo.com/v1/forecast"
    params={"latitude":lat,"longitude":lon,"current":"temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,showers,weather_code,wind_speed_10m,wind_gusts_10m,visibility","timezone":"Asia/Kolkata","forecast_days":1}
    data,err=http_get(url,params)
    if err: return None,err
    return data,None

def weather_name(code):
    if code is None: return "Unknown"
    c=int(code)
    mapping={0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Fog",48:"Rime fog",
             51:"Light drizzle",53:"Drizzle",55:"Dense drizzle",61:"Slight rain",63:"Rain",65:"Heavy rain",
             71:"Slight snow",73:"Snow",75:"Heavy snow",80:"Rain showers",81:"Rain showers",82:"Violent rain showers",
             95:"Thunderstorm",96:"Thunderstorm with hail",99:"Thunderstorm with hail"}
    return mapping.get(c,"Weather code "+str(c))

def geocode_places():
    # Curated Andhra Pradesh cities/towns to avoid dependence on geocoding service.
    return [
      ("Visakhapatnam",17.6868,83.2185),("Vijayawada",16.5062,80.6480),("Guntur",16.3067,80.4365),
      ("Tirupati",13.6288,79.4192),("Nellore",14.4426,79.9865),("Kurnool",15.8281,78.0373),
      ("Rajahmundry",17.0005,81.8040),("Kakinada",16.9891,82.2475),("Anantapur",14.6819,77.6006),
      ("Kadapa",14.4673,78.8242),("Ongole",15.5057,80.0499),("Srikakulam",18.2949,83.8938),
      ("Eluru",16.7107,81.0952),("Machilipatnam",16.1809,81.1300),("Chittoor",13.2172,79.1003),
      ("Amaravati",16.5745,80.3598)
    ]

@st.cache_data(ttl=300, show_spinner=False)
def fetch_osm_road_context():
    # Overpass is open community infrastructure; returns road geometry, NOT live traffic speed/incidents.
    south,north,west,east=AP_BBOX
    query=f"""[out:json][timeout:25];(way[highway~"motorway|trunk|primary|secondary"]( {south},{west},{north},{east}););out center tags;"""
    try:
        r=requests.post("https://overpass-api.de/api/interpreter",data=query,timeout=35,
                        headers={"User-Agent":"SmartTrafficAPAcademicProject/1.0"})
        r.raise_for_status()
        return r.json(),None
    except Exception as e:
        return None,str(e)


@st.cache_data(ttl=180, show_spinner=False)
def fetch_tomtom_incidents(api_key, lat, lon):
    # TomTom Incident Details v5. Query a city-sized bbox, not the entire state.
    delta=0.12
    bbox=f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
    url="https://api.tomtom.com/traffic/services/5/incidentDetails"
    params={"key":api_key,"bbox":bbox,"language":"en-GB","timeValidityFilter":"present"}
    return http_get(url,params,timeout=20)

@st.cache_data(ttl=180, show_spinner=False)
def fetch_tomtom_flow(api_key, lat, lon):
    url="https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params={"key":api_key,"point":f"{lat},{lon}","unit":"KMPH","thickness":3,"openLr":False}
    return http_get(url,params,timeout=20)

def tomtom_incident_rows(payload):
    rows=[]
    for incident in (payload or {}).get("incidents",[]):
        p=incident.get("properties",{}) or {}
        geom=incident.get("geometry",{}) or {}
        coords=geom.get("coordinates",[])
        if geom.get("type")=="Point" and len(coords)>=2:
            ilon,ilat=coords[0],coords[1]
        elif geom.get("type")=="LineString" and coords:
            ilon=sum(pt[0] for pt in coords)/len(coords); ilat=sum(pt[1] for pt in coords)/len(coords)
        else: ilat=ilon=None
        events=p.get("events",[]) or []
        desc="; ".join(e.get("description","") for e in events if e.get("description"))
        category={1:"Accident",2:"Fog",3:"Dangerous conditions",4:"Rain",5:"Ice",6:"Traffic jam",7:"Lane closed",
                  8:"Road closed",9:"Road works",10:"Wind",11:"Flooding",14:"Broken-down vehicle"}.get(p.get("iconCategory"),"Traffic incident")
        rows.append({"Type":category,"Description":desc or category,"Road":", ".join(p.get("roadNumbers",[]) or []),
                     "From":p.get("from",""),"To":p.get("to",""),"Delay (min)":round((p.get("delay") or 0)/60,1),
                     "Severity":{0:"Unknown",1:"Minor",2:"Moderate",3:"Major",4:"Undefined"}.get(p.get("magnitudeOfDelay"),"Unknown"),
                     "Last report":p.get("lastReportTime",""),"Latitude":ilat,"Longitude":ilon,
                     "Probability":p.get("probabilityOfOccurrence",""),"ID":p.get("id","")})
    return rows

def load_training():
    if not DATA_FILE.exists(): return None,None,None,None
    df=pd.read_csv(DATA_FILE,low_memory=False)
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError("CSV missing columns: "+", ".join(missing))
    df=df.drop_duplicates().copy()
    df["Start_Time"]=pd.to_datetime(df["Start_Time"],errors="coerce")
    for c in ["Severity","Start_Lat","Start_Lng"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["Weather_Condition"]=df["Weather_Condition"].fillna("Unknown").astype(str)
    df=df.dropna(subset=["Severity","Start_Time","Start_Lat","Start_Lng"])
    df=df[df["Start_Lat"].between(-90,90)&df["Start_Lng"].between(-180,180)]
    df["Severity"]=df["Severity"].astype(int)
    df["Hour"]=df.Start_Time.dt.hour; df["Day_Number"]=df.Start_Time.dt.dayofweek
    df["Month"]=df.Start_Time.dt.month; df["Year"]=df.Start_Time.dt.year
    df["Is_Weekend"]=(df.Day_Number>=5).astype(int)
    enc=LabelEncoder(); df["Weather_Encoded"]=enc.fit_transform(df.Weather_Condition)
    X=df[FEATURES]; y=df.Severity
    strat=y if y.value_counts().min()>=2 else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=strat)
    model=RandomForestClassifier(n_estimators=150,random_state=42,class_weight="balanced",n_jobs=-1)
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    metrics={"accuracy":accuracy_score(yte,pred),"cm":confusion_matrix(yte,pred,labels=model.classes_),
             "classes":model.classes_,"report":classification_report(yte,pred,output_dict=True,zero_division=0)}
    joblib.dump(model,MODEL_FILE); joblib.dump(enc,ENCODER_FILE); joblib.dump(metrics,METRICS_FILE)
    return df,model,enc,metrics

def risk_from_weather(w):
    if not w: return "Unavailable"
    c=w.get("current",{})
    rain=float(c.get("precipitation",0) or 0); gust=float(c.get("wind_gusts_10m",0) or 0)
    vis=float(c.get("visibility",10000) or 10000); code=int(c.get("weather_code",0) or 0)
    if rain>=5 or code in [65,82,95,96,99] or vis<1000: return "Elevated"
    if rain>0 or gust>=45 or code in [45,48,61,63,80,81]: return "Moderate"
    return "Lower"

def city_weather_panel(city,lat,lon):
    w,err=get_weather(lat,lon)
    if err:
        st.warning(f"{city}: weather feed unavailable ({err})"); return
    c=w.get("current",{})
    st.markdown(f"**{city}**")
    st.metric("Temperature",f"{c.get('temperature_2m','—')} °C",f"Feels like {c.get('apparent_temperature','—')} °C")
    st.caption(f"{weather_name(c.get('weather_code'))} • Rain {c.get('precipitation','—')} mm • Wind {c.get('wind_speed_10m','—')} km/h • Visibility {c.get('visibility','—')} m")
    st.caption("Weather-derived driving condition indicator: "+risk_from_weather(w))

def build_map(city_name, city_lat, city_lon, weather):
    m=folium.Map(location=[AP_CENTER[0],AP_CENTER[1]],zoom_start=7,tiles="CartoDB dark_matter")
    for name,lat,lon in geocode_places():
        popup=f"<b>{name}</b><br>Live weather is shown in dashboard city panels."
        folium.CircleMarker([lat,lon],radius=5,color="#65d9cb",fill=True,fill_opacity=.8,popup=popup,tooltip=name).add_to(m)
    if city_name and city_lat and city_lon:
        color={"Elevated":"red","Moderate":"orange","Lower":"green"}.get(risk_from_weather(weather),"blue")
        folium.Marker([city_lat,city_lon],tooltip=f"{city_name} selected",
                      popup=f"{city_name}: weather-derived condition {risk_from_weather(weather)}",
                      icon=folium.Icon(color=color,icon="info-sign")).add_to(m)
    return m

st.markdown('<div class="hero"><div class="kicker">ANDHRA PRADESH • LIVE DATA LAYER</div><h1>Smart Traffic Accident Intelligence</h1><div class="sub">Live weather conditions, open road-network context, reported-incident integration point and historical ML analysis.</div></div>',unsafe_allow_html=True)
st.sidebar.title("🚦 Smart Traffic AP")
st.sidebar.caption("Live intelligence dashboard")
page=st.sidebar.radio("NAVIGATION",["Live Overview","Live Weather","Traffic & Road Network","Incident Feed","Historical Analytics","Severity Prediction","Model Performance"])
st.sidebar.divider()
city_options=[x[0] for x in geocode_places()]
selected_city=st.sidebar.selectbox("Focus city",city_options,index=1)
city_lookup={x[0]:(x[1],x[2]) for x in geocode_places()}
lat,lon=city_lookup[selected_city]
st.sidebar.markdown("### Optional live traffic API")
tomtom_key=st.sidebar.text_input("TomTom API key",type="password",help="Optional: enables current incidents and a nearby traffic-flow speed segment. Kept in this Streamlit session only.")
st.sidebar.caption("Without a key, live weather remains available; traffic flow and provider incident data are disabled.")
refresh=st.sidebar.button("↻ Refresh live feeds")
if refresh:
    get_weather.clear(); fetch_osm_road_context.clear(); fetch_tomtom_incidents.clear(); fetch_tomtom_flow.clear(); st.rerun()
st.sidebar.caption("Weather cache: 5 minutes • Road network: 5 minutes")

with st.spinner("Fetching current weather..."):
    current_weather, weather_error=get_weather(lat,lon)

if page=="Live Overview":
    st.subheader("Andhra Pradesh live snapshot")
    c1,c2,c3,c4=st.columns(4)
    cur=(current_weather or {}).get("current",{})
    with c1: metric("FOCUS CITY",selected_city,"Selected monitoring location")
    with c2: metric("TEMPERATURE",f"{cur.get('temperature_2m','—')} °C","Open-Meteo current observation")
    with c3: metric("PRECIPITATION",f"{cur.get('precipitation','—')} mm","Current reported precipitation")
    with c4: metric("WEATHER ROAD INDICATOR",risk_from_weather(current_weather),"Weather proxy, not accident probability")
    st.info("Live-data honesty: public free sources provide weather and road-network context. A verified, comprehensive real-time accident feed for all Andhra Pradesh is not assumed. The incident panel below accepts official/authorized feeds or manual verified reports.")
    left,right=st.columns([1.15,1])
    with left:
        st.markdown("### Andhra Pradesh monitoring map")
        st_folium(build_map(selected_city,lat,lon,current_weather),height=540,use_container_width=True,returned_objects=[])
    with right:
        st.markdown("### Current weather")
        if weather_error: st.error(weather_error)
        else:
            st.metric("Temperature",f"{cur.get('temperature_2m','—')} °C",f"Feels like {cur.get('apparent_temperature','—')} °C")
            st.metric("Wind",f"{cur.get('wind_speed_10m','—')} km/h",f"Gusts {cur.get('wind_gusts_10m','—')} km/h")
            st.metric("Visibility",f"{cur.get('visibility','—')} m")
            st.write("**Condition:**",weather_name(cur.get("weather_code")))
            st.write("**Weather-derived road indicator:**",risk_from_weather(current_weather))
            st.caption("This is a weather-based driving-condition indicator, not a prediction that an accident will occur.")
        st.markdown("### Live data status")
        st.success("Open-Meteo weather API: connected" if not weather_error else "Weather API unavailable")
        if tomtom_key:
            incident_payload, incident_err=fetch_tomtom_incidents(tomtom_key,lat,lon)
            if incident_err: st.error("TomTom incident API: "+incident_err)
            else: st.success(f"TomTom incident feed connected • {len(tomtom_incident_rows(incident_payload))} incidents in selected-city search area")
        else:
            st.warning("TomTom live traffic/incident feed: add an API key in the sidebar")
        st.info("OpenStreetMap road network is map context only; it does not provide live traffic speeds.")



elif page=="Live Weather":
    st.subheader("Current weather across Andhra Pradesh")
    st.caption("Current conditions are fetched from Open-Meteo. Results are cached for five minutes.")

    cols = st.columns(2)
    for i, (name, clat, clon) in enumerate(geocode_places()):
        with cols[i % 2]:
            with st.container(border=True):
                city_weather_panel(name, clat, clon)

elif page=="Traffic & Road Network":
    st.subheader("Live traffic flow & road network")
    if tomtom_key:
        flow,flow_err=fetch_tomtom_flow(tomtom_key,lat,lon)
        if flow_err: st.error("TomTom flow API: "+flow_err)
        else:
            seg=flow.get("flowSegmentData",{})
            f1,f2,f3=st.columns(3)
            with f1: metric("CURRENT SPEED",f"{seg.get('currentSpeed','—')} km/h","TomTom observed traffic speed")
            with f2: metric("FREE-FLOW SPEED",f"{seg.get('freeFlowSpeed','—')} km/h","Reference speed")
            with f3: metric("CONFIDENCE",f"{float(seg.get('confidence',0))*100:.0f}%","Provider confidence score")
            if seg.get("currentSpeed") is not None and seg.get("freeFlowSpeed"):
                ratio=float(seg["currentSpeed"])/max(float(seg["freeFlowSpeed"]),1)
                st.info("Traffic flow indicator: "+("Heavy congestion" if ratio<0.4 else "Slow traffic" if ratio<0.7 else "Near free flow"))
            st.caption("One queried road segment near the selected city; not a statewide traffic average.")
    else:
        st.warning("Add a TomTom API key in the sidebar to enable live observed speed and traffic flow.")
    st.markdown("### OpenStreetMap road network context")
    with st.spinner("Querying OpenStreetMap Overpass road data..."):
        osm,err=fetch_osm_road_context()
    if err:
        st.error("Overpass request failed: "+err)
    else:
        elements=osm.get("elements",[])
        roads=[]
        for e in elements:
            tags=e.get("tags",{}); center=e.get("center",{})
            if center:
                roads.append({"Road":tags.get("name","Unnamed road"),"Class":tags.get("highway","unknown"),
                              "Latitude":center.get("lat"),"Longitude":center.get("lon")})
        st.metric("Mapped major road segments returned",f"{len(roads):,}")
        if roads:
            st.dataframe(pd.DataFrame(roads).head(500),use_container_width=True,hide_index=True)
        st.caption("Source: OpenStreetMap contributors via Overpass API. Coverage and availability depend on community mapping and public service load.")

elif page=="Incident Feed":
    st.subheader("Live traffic incidents & reported accidents")
    st.caption(f"Search area: around {selected_city}, Andhra Pradesh. Provider results depend on coverage and selected bounding box; this is not statewide coverage.")
    if tomtom_key:
        payload,err=fetch_tomtom_incidents(tomtom_key,lat,lon)
        if err: st.error("TomTom incident API request failed: "+err)
        else:
            rows=tomtom_incident_rows(payload)
            if rows:
                st.success(f"{len(rows)} current provider-reported incidents returned for this city search area.")
                st.dataframe(pd.DataFrame(rows).drop(columns=["Latitude","Longitude","ID"]),use_container_width=True,hide_index=True)
                imap=folium.Map(location=[lat,lon],zoom_start=11,tiles="CartoDB dark_matter")
                for item in rows:
                    if item["Latitude"] is not None and item["Longitude"] is not None:
                        folium.Marker([item["Latitude"],item["Longitude"]],popup=f"{item['Type']}: {item['Description']}",
                          tooltip=item["Type"],icon=folium.Icon(color="red" if item["Type"]=="Accident" else "orange")).add_to(imap)
                st_folium(imap,height=430,use_container_width=True,returned_objects=[])
            else: st.info("No current incidents returned for this selected-city search area. This does not prove none exist.")
    else:
        st.warning("For automatic live accident/incident reports, add a TomTom API key in the sidebar. The app does not invent incidents.")
        st.markdown("TomTom's Traffic API provides current incident details and observed traffic flow. Create a developer key and check plan, quota, India coverage and terms.")
    st.markdown("### Add a verified incident manually")
    with st.form("incident"):
        a,b=st.columns(2)
        with a:
            incident_city=st.selectbox("City / location",city_options)
            incident_type=st.selectbox("Incident type",["Accident","Road obstruction","Vehicle breakdown","Flooding / waterlogging","Other"])
            incident_date=st.date_input("Reported date", value=datetime.now().date())
            incident_clock=st.time_input("Reported time", value=datetime.now().time().replace(second=0, microsecond=0))
            incident_time=datetime.combine(incident_date, incident_clock)
        with b:
            severity=st.selectbox("Reported severity",["Unconfirmed","Minor","Major","Critical"])
            source=st.text_input("Source / reference (required)",placeholder="Official notice, police update, verified source URL")
            details=st.text_area("Details")
        submitted=st.form_submit_button("Save incident locally",type="primary")
    if submitted:
        if not source.strip():
            st.error("Please provide a source/reference. Unverified reports should not be presented as confirmed.")
        else:
            file=ROOT/"reported_incidents.csv"
            new=pd.DataFrame([{"reported_at":incident_time.isoformat(),"city":incident_city,"type":incident_type,
                               "severity":severity,"source":source,"details":details,"status":"User-entered; not independently verified"}])
            new.to_csv(file,mode="a",header=not file.exists(),index=False)
            st.success("Incident saved locally. It is labelled as user-entered and not independently verified.")
    file=ROOT/"reported_incidents.csv"
    if file.exists():
        st.markdown("### Locally recorded reports")
        st.dataframe(pd.read_csv(file),use_container_width=True,hide_index=True)
        st.download_button("Download incident reports CSV",file.read_bytes(),"reported_incidents.csv","text/csv")
    else:
        st.caption("No incident reports have been entered in this project yet.")

elif page=="Historical Analytics":
    st.subheader("Historical accident analytics")
    if not DATA_FILE.exists():
        st.info("Add traffic_accident.csv to enable historical analytics and ML training. Live weather pages work without it.")
    else:
        try:
            df,model,enc,metrics=load_training()
            a,b,c=st.columns(3)
            with a: metric("HISTORICAL RECORDS",f"{len(df):,}","From local CSV")
            with b: metric("TEST ACCURACY",f"{metrics['accuracy']*100:.1f}%","Random Forest hold-out test")
            with c: metric("SEVERITY CLASSES",df.Severity.nunique(),"In historical dataset")
            x,y=st.columns(2)
            with x:
                counts=df.Severity.value_counts().sort_index().rename_axis("Severity").reset_index(name="Accidents")
                st.plotly_chart(px.bar(counts,x="Severity",y="Accidents",color="Severity",title="Accidents by severity"),use_container_width=True)
            with y:
                hourly=df.groupby("Hour").size().reindex(range(24),fill_value=0).rename("Accidents").reset_index()
                st.plotly_chart(px.line(hourly,x="Hour",y="Accidents",markers=True,title="Historical accidents by hour"),use_container_width=True)
            st.caption("These are historical CSV-derived analytics, not live incident counts.")
        except Exception as e: st.error(str(e))

elif page=="Severity Prediction":
    st.subheader("Historical ML severity estimate")
    st.caption("Prediction uses the locally supplied historical dataset. It is not a live accident alert.")
    if not DATA_FILE.exists():
        st.info("Add traffic_accident.csv to train the model and enable prediction.")
    else:
        try:
            df,model,enc,metrics=load_training()
            opts=sorted(enc.classes_.tolist())
            with st.form("predict"):
                d1,d2,d3=st.columns(3)
                with d1: day=st.date_input("Date")
                with d2: hour=st.slider("Hour",0,23,17)
                with d3: weather=st.selectbox("Weather",opts)
                p1,p2=st.columns(2)
                with p1: plat=st.number_input("Latitude",value=float(df.Start_Lat.median()),format="%.6f")
                with p2: plon=st.number_input("Longitude",value=float(df.Start_Lng.median()),format="%.6f")
                go=st.form_submit_button("Estimate severity",type="primary")
            if go:
                dn=pd.Timestamp(day).dayofweek
                row=pd.DataFrame([{"Hour":hour,"Day_Number":dn,"Month":day.month,"Year":day.year,"Is_Weekend":int(dn>=5),
                                   "Start_Lat":plat,"Start_Lng":plon,"Weather_Encoded":int(enc.transform([weather])[0])}],columns=FEATURES)
                pred=int(model.predict(row)[0]); probs=model.predict_proba(row)[0]
                st.success(f"Predicted historical severity class: {pred}")
                st.metric("Highest class probability",f"{max(probs)*100:.1f}%")
                st.plotly_chart(px.bar(pd.DataFrame({"Severity":model.classes_.astype(str),"Probability":probs}),x="Severity",y="Probability",title="Model class probabilities"),use_container_width=True)
                st.warning("Not for emergency response or operational road-safety decisions.")
        except Exception as e: st.error(str(e))

elif page=="Model Performance":
    st.subheader("Historical model performance")
    if not DATA_FILE.exists():
        st.info("Add traffic_accident.csv to train and evaluate the model.")
    else:
        try:
            df,model,enc,metrics=load_training()
            st.metric("Hold-out accuracy",f"{metrics['accuracy']*100:.2f}%")
            st.plotly_chart(px.imshow(metrics["cm"],text_auto=True,x=[str(x) for x in metrics["classes"]],
                y=[str(x) for x in metrics["classes"]],labels={"x":"Predicted","y":"Actual","color":"Count"},
                title="Confusion matrix"),use_container_width=True)
            imp=pd.DataFrame({"Feature":FEATURES,"Importance":model.feature_importances_}).sort_values("Importance")
            st.plotly_chart(px.bar(imp,x="Importance",y="Feature",orientation="h",title="Feature importance"),use_container_width=True)
        except Exception as e: st.error(str(e))

st.sidebar.divider()
st.sidebar.caption("Live weather: Open-Meteo • Road data: OpenStreetMap")
st.sidebar.caption("Accident feed: requires a verified provider or official integration")
