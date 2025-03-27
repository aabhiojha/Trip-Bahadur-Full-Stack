import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polyline,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import marker from "../../assets/map-marker-svgrepo-com.svg";

const redOptions = { color: "red" };

const myIcon = new L.Icon({
  iconUrl: marker,
  iconRetinaUrl: marker,
  iconSize: [32, 45],
  popupAnchor: [0, -10],
});

const MapComponent = ({ routeData }) => {
  if (!routeData || routeData.length === 0)
    return <p>No route data available</p>;

  // Generate polyline path
  const polyData = routeData.reduce((acc, point, index, arr) => {
    if (index < arr.length - 1) {
      acc.push([
        [point.coordinates.latitude, point.coordinates.longitude],
        [
          arr[index + 1].coordinates.latitude,
          arr[index + 1].coordinates.longitude,
        ],
      ]);
    }
    return acc;
  }, []);
  console.log(polyData);
  return (
    <MapContainer
      className="map-container w-[800px] h-[600px] rounded-2xl"
      center={[28.2403, 83.9856]}
      zoom={7}
      scrollWheelZoom={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        // url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        // url = "https://tiles.stadiamaps.com/tiles/alidade_satellite/{z}/{x}/{y}{r}.png"
        url = "https://tiles.stadiamaps.com/tiles/alidade_satellite/{z}/{x}/{y}{r}.png"
      />

      {routeData.map(({ day, location, highlight, elevation, coordinates }, index) => (
        <Marker
          key={`${location}-${day}-${index}`}
          position={[coordinates.latitude, coordinates.longitude]}
          icon={myIcon}
        >
          <Popup>
            <strong>Day:</strong> {day}
            <br />
            <strong>Location:</strong> {location}
            <br />
            <strong>Elevation:</strong> {elevation} meters
            <br />
            <strong>Highlight:</strong> {highlight}
            <br />
            <strong>Coordinates:</strong> {coordinates.latitude},{" "}
            {coordinates.longitude}
          </Popup>
        </Marker>
      ))}
      <Polyline pathOptions={redOptions} positions={polyData} />
    </MapContainer>
  );
};

export default MapComponent;
