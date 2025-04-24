import React from "react";
import CandlestickChart from "./components/CandlestickChart";

function App() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", padding: "1rem", background: "#111", color: "#fff" }}>
      <div>
        <h3>1 Hour Chart</h3>
        <CandlestickChart interval="1h" />
      </div>
      <div>
        <h3>4 Hour Chart</h3>
        <CandlestickChart interval="4h" />
      </div>
      <div>
        <h3>1 Day Chart</h3>
        <CandlestickChart interval="1d" />
      </div>
      <div>
        <h3>1 Week Chart</h3>
        <CandlestickChart interval="1w" />
      </div>
    </div>
  );
}

export default App;
