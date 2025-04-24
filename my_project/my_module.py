import React, { useEffect, useState } from "react";
import { createChart } from "lightweight-charts";
import axios from "axios";

const chartConfig = {
  width: 400,
  height: 300,
  layout: {
    background: { color: "#ffffff" },
    textColor: "#000" 
  },
  grid: {
    vertLines: { color: "#eee" },
    horzLines: { color: "#eee" }
  },
  priceScale: {
    borderColor: "#ccc"
  },
  timeScale: {
    borderColor: "#ccc",
    timeVisible: true,
    secondsVisible: false
  }
};

const timeframes = [
  { label: "1h", interval: "1h" },
  { label: "4h", interval: "4h" },
  { label: "1d", interval: "1d" },
  { label: "1w", interval: "1w" }
];

const CandlestickChart = ({ interval }) => {
  const chartContainerRef = React.useRef();
  const [chart, setChart] = useState(null);

  const fetchCandles = async () => {
    try {
      const response = await axios.get(`/api/candles?interval=${interval}`);
      const candleSeries = chart.addCandlestickSeries();
      candleSeries.setData(response.data);
    } catch (err) {
      console.error("Error fetching candles:", err);
    }
  };

  useEffect(() => {
    const newChart = createChart(chartContainerRef.current, chartConfig);
    setChart(newChart);

    fetchCandles();
    const intervalId = setInterval(fetchCandles, 60000);

    return () => {
      newChart.remove();
      clearInterval(intervalId);
    };
  }, []);

  return <div ref={chartContainerRef} />;
};

export default function Dashboard() {
  return (
    <div className="grid grid-cols-2 gap-4 p-4">
      {timeframes.map((tf) => (
        <div key={tf.label} className="border rounded-xl shadow p-2">
          <h2 className="text-xl font-bold mb-2 text-center">{tf.label} Chart</h2>
          <CandlestickChart interval={tf.interval} />
        </div>
      ))}
    </div>
  );
}
