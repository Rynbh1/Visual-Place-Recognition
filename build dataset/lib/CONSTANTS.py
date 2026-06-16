# ---------------------------------------------------------------------------
# Constantes Mapillary
# ---------------------------------------------------------------------------
TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"
GRAPH_URL = "https://graph.mapillary.com/{id}"
COVERAGE_ZOOM = 14  # zoom auquel la couche "image" est disponible
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Taille moyenne d'une image JPEG selon la resolution (Mo) -> pour l'estimation
AVG_MB = {"256": 0.02, "1024": 0.18, "2048": 0.55, "original": 2.5}
THUMB_FIELD = {
    "256": "thumb_256_url",
    "1024": "thumb_1024_url",
    "2048": "thumb_2048_url",
    "original": "thumb_original_url",
}

# ---------------------------------------------------------------------------
# Supplement Panoramax / KartaView + carte de couverture
# ---------------------------------------------------------------------------
MIN_PHOTOS_PER_PLACE = 4
PANORAMAX_SEARCH = "https://panoramax.ign.fr/api/search"
KARTAVIEW_PHOTOS = "https://api.kartaview.org/2.0/photo/"
AVG_CROP_MB = 0.20


# Template HTML Leaflet — les placeholders LAT_CTR/LON_CTR/GEOJSON/MIN_P sont
# substitues par str.replace() pour eviter tout conflit avec les accolades JS.
_COVERAGE_MAP_TMPL = """\
<!DOCTYPE html><html><head>
<meta charset="utf-8"><title>Couverture street-level</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{margin:0}#map{height:100vh}</style>
</head><body><div id="map"></div><script>
var map=L.map("map").setView([LAT_CTR,LON_CTR],15);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",{attribution:"&copy; OpenStreetMap"}).addTo(map);
var gj=GEOJSON;
L.geoJSON(gj,{
  style:function(f){return{color:f.properties.color,weight:1,fillOpacity:0.55,fillColor:f.properties.color};},
  onEachFeature:function(f,layer){layer.bindPopup(f.properties.popup);}
}).addTo(map);
var leg=L.control({position:"bottomright"});
leg.onAdd=function(){
  var d=L.DomUtil.create("div");
  d.style.cssText="background:white;padding:8px;border-radius:4px;font-size:13px";
  d.innerHTML="<b>Couverture</b><br>"
    +"<span style='color:#2ecc71'>&#9632;</span> Mapillary &ge;MIN_P<br>"
    +"<span style='color:#27ae60'>&#9632;</span> Complément &ge;MIN_P<br>"
    +"<span style='color:#e67e22'>&#9632;</span> Insuffisant<br>"
    +"<span style='color:#e74c3c'>&#9632;</span> Absent";
  return d;
};
leg.addTo(map);
</script></body></html>"""
