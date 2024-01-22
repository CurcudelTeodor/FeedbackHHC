import "./App.css";
import { useState } from "react";
import Chart from "./components/Chart";
import AgencySearch from "./components/AgencySearch";
import Predict from "./components/Predict";
function App() {
  // TODO: flip
  // const [component, setComponent] = useState(<Chart />);
  const [component, setComponent] = useState(<Predict />);

  return (
    <>
      <div className="buttons">
        <button onClick={() => setComponent(<Chart />)}>Charts</button>
        <button onClick={() => setComponent(<AgencySearch />)}>Agencies</button>
        <button onClick={() => setComponent(<Predict />)}>Predict</button>
      </div>

      <div className="container">
        {component}
      </div>
    </>
  );
}

export default App;
