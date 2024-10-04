import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function App() {
  const [currentAQI, setCurrentAQI] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    // Fetch current AQI
    fetch('http://localhost:5000/api/current')
      .then(response => response.json())
      .then(data => setCurrentAQI(data));

    // Fetch historical data
    fetch('http://localhost:5000/api/historical')
      .then(response => response.json())
      .then(data => setHistoricalData(data));

    // Fetch prediction
    fetch('http://localhost:5000/api/predict')
      .then(response => response.json())
      .then(data => setPrediction(data));
  }, []);

  return (
    <div className="App">
      <h1>Air Quality Dashboard</h1>
      
      <h2>Current AQI</h2>
      {currentAQI && (
        <p>
          Current AQI: {currentAQI.value.toFixed(2)} (as of {new Date(currentAQI.timestamp).toLocaleString()})
        </p>
      )}

      <h2>AQI Prediction</h2>
      {prediction && (
        <p>
          Predicted AQI: {prediction.value.toFixed(2)} (for {new Date(prediction.timestamp).toLocaleString()})
        </p>
      )}

      <h2>Historical AQI Data</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={historicalData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="timestamp" 
            tickFormatter={(timestamp) => new Date(timestamp).toLocaleDateString()}
          />
          <YAxis />
          <Tooltip 
            labelFormatter={(label) => new Date(label).toLocaleString()}
            formatter={(value) => value.toFixed(2)}
          />
          <Legend />
          <Line type="monotone" dataKey="value" stroke="#8884d8" name="AQI" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default App;