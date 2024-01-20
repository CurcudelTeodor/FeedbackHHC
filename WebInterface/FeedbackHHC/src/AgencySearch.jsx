import { useEffect, useState } from "react";
const AgencySearch = () => {
  const [agencyName, setAgencyName] = useState("");
  const [agencies, setAgencies] = useState([]);
  const handleNameChange = (event) => {
    setAgencyName(event.target.value);
  };
  const fetchAgencies = () => {
    fetch("http://127.0.0.1:5000/agencies")
      .then((res) => res.json())
      .then((data) => {
        console.log(data.agencies); 
        setAgencies(data.agencies); 
      })
      .catch((error) => {
        console.error("Error fetching agencies:", error);
      });
  };
  
  useEffect(() => {
    fetchAgencies();
  }, []);
  return (
    <div>
      <label>
        Agency Name: <input name="myInput" onChange={handleNameChange} />
      </label>
      <button>Submit</button>
      <p>{agencyName}</p>
    </div>
  );
};

export default AgencySearch;
