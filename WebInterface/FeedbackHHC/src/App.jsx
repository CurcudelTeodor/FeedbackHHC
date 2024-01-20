import Chart from "./Chart";
import "./App.css";
import { useState } from "react";
import AgencySearch from "./AgencySearch";
function App() {
  const [selectedButton, setSelectedButton] = useState("1");
  return (
    <>
      <div className="buttons">
        <button onClick={() => setSelectedButton("1")}>Charts</button>
        <button onClick={() => setSelectedButton("2")}>Agencies</button>
      </div>
      {selectedButton === "1" && <Chart />}
      {selectedButton === "2" && (
        <div className="container">
          <AgencySearch />
        </div>
      )}
    </>
  );
}

export default App;
