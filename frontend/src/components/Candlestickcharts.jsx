import React, { useEffect, useRef, useState } from "react";
import { createChart } from "lightweight-charts";
import axios from "axios";

const INTERVAL_MAP = {
  "1h": "1h",
  "4h": "4h",
  "1d": "1d",
  "1w": "1w",
};

function CandlestickChart({ interval }) {
  const chartContainerRef = useRef();
  const [chart, setChart] = useState(null);

  useEffect(() => {
    // Initialize chart only once
    const chartInstance = createChart(chartContainerRef.current, {
      width: 400,
      height: 300,
      layout: { backgroundColor: "#000", textColor: "#fff" },
      grid: { vertLines: { color: "#444" }, horzLines: { color: "#444" } },
    });

    const candleSeries = chartInstance.addCandlestickSeries();
    setChart(chartInstance);

    // Function to fetch candlestick data
    const fetchCandles = async () => {
      try {
        const response = await axios.get(
          `https://opulent-yodel-vx676gv56vr39qp-8000.app.github.dev/candles/XRPUSDT/${INTERVAL_MAP[interval]}`
        );
        const data = response.data.map(candle => ({
          time: Math.floor(candle.time / 1000), // Convert time to UNIX timestamp (seconds)
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
        }));
        candleSeries.setData(data);
      } catch (error) {
        console.error("Error fetching candle data:", error);
      }
    };

    // Initial data fetch
    fetchCandles();

    // Set interval to update chart every minute
    const intervalId = setInterval(fetchCandles, 60000); // 60000ms = 1 minute

    // Cleanup on component unmount
    return () => {
      clearInterval(intervalId);
      chartInstance.remove();
    };
  }, [interval]);

  return <div ref={chartContainerRef} />;
}

export default CandlestickChart;
