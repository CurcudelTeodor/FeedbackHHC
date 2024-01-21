import { useEffect, useState } from "react";
import { ToastContainer, toast } from "react-toastify";
import { FaCheck,FaRegWindowClose  } from "react-icons/fa";
import "react-toastify/dist/ReactToastify.css";
import StarRating from "./StarRating";
const AgencySearch = () => {
  const [agencyName, setAgencyName] = useState("");
  const [zipCode, setZipCode] = useState("");
  const [agencies, setAgencies] = useState([]);
  const [agencyData, setAgencyData] = useState(null);
  

  const handleNameChange = (event) => {
    setAgencyName(event.target.value);
  };
  const handleZipCodeChange = (event) => {
    setZipCode(event.target.value);
  };
  const handleNameSet = (agency) => {
    setAgencyName(agency);
  };
  const fetchAgencies = () => {
    fetch("http://127.0.0.1:5000/agencies")
      .then((res) => res.json())
      .then((data) => {
        setAgencies(data.agencies);
      })
      .catch((error) => {
        toast.error(error);
      });
  };
  const onSearch = (searchTerm) => {
    setAgencyName(searchTerm);
    if (searchTerm === "" || zipCode === "") {
      toast.error("Please enter a valid agency name and zip code");
      return;
    }
    fetch(`http://127.0.0.1:5000/agency/${searchTerm}/${zipCode}`)
      .then((res) => res.json())
      .then((data) => {
        if (data.error) {
          toast.error(data.error);
          return;
        }
        const newData = {
          Address: data[0].Address,
          "City/Town": data[0]["City/Town"],
          State: data[0].State,
          "Telephone Number": data[0]["Telephone Number"],
          "Provider Name": data[0]["Provider Name"],
          "Offers Home Health Aide Services":
            data[0]["Offers Home Health Aide Services"],
          "Offers Medical Social Services":
            data[0]["Offers Medical Social Services"],
          "Offers Nursing Care Services":
            data[0]["Offers Nursing Care Services"],
          "Offers Occupational Therapy Services":
            data[0]["Offers Occupational Therapy Services"],
          "Offers Physical Therapy Services":
            data[0]["Offers Physical Therapy Services"],
          "Offers Speech Pathology Services":
            data[0]["Offers Speech Pathology Services"],
          "ZIP Code": data[0]["ZIP Code"],
          "Quality of patient care star rating":
            data[0]["Quality of patient care star rating"],
        };
        setAgencyData(newData);
      })
      .catch((error) => {
        toast.error(error);
      });
  };
  useEffect(() => {
    fetchAgencies();
  }, []);
  return (
    <div>
      <div className="search-page">
        <div className="search-container">
          <div className="search-inner">
            <input
              type="text"
              value={agencyName}
              onChange={handleNameChange}
              placeholder="Enter agency name"
            />
            <input
              type="text"
              value={zipCode}
              onChange={handleZipCodeChange}
              placeholder="Enter agency zip code"
            />
            <button className="search-btn" onClick={() => onSearch(agencyName)}>
              Search
            </button>
          </div>
          <div className="dropdown">
            {agencies
              .filter((item) => {
                const searchTerm = agencyName.toLowerCase();
                const agency = item.toLowerCase();

                return (
                  searchTerm &&
                  agency.startsWith(searchTerm) &&
                  (agencyName === "" || agency !== searchTerm)
                );
              })
              .slice(0, 10)
              .map((item) => (
                <div
                  onClick={() => handleNameSet(item)}
                  className="dropdown-row"
                  key={item}
                >
                  {item}
                </div>
              ))}
          </div>
        </div>
        {agencyData && (
          <div className="search-results">
            <div className="top-information">
              <div className="agency-name">
                <h2>Home health agency</h2>
                <h1>{agencyData["Provider Name"]}</h1>
              </div>
              <div className="information">
                <p className="bold">OFFICE LOCATION</p>
                <p>{agencyData.Address}</p>
                <p>
                  {agencyData["City/Town"]}, {agencyData.State}{" "}
                  {agencyData["ZIP Code"]}
                </p>
                <p className="bold">PHONE NUMBER</p>
                <p>{agencyData["Telephone Number"]}</p>
              </div>
            </div>
            <div className="star-rating">
              <p>Quality of patient care star rating</p>
              <StarRating
                rating={agencyData["Quality of patient care star rating"]}
              />
            </div>
            <div className="services">
              <div>
                <h3>Services Provided</h3>
              </div>
              <div>
                <ul>
                  <li className="list-item">
                    {"Offers Home Health Aide Services"}
                    <div className="services-item">
                    {agencyData["Offers Home Health Aide Services"] === "Yes" ? <FaCheck color="green"/>   : <FaRegWindowClose color="red"/>}
                    {agencyData["Offers Home Health Aide Services"]}
                    </div>
                  </li>
                  <li className="list-item">
                    {"Offers Medical Social Services"}
                    <div className="services-item">
                    {agencyData["Offers Medical Social Services"] === "Yes" ? <FaCheck color="green"/>   : <FaRegWindowClose color="red"/>}
                    {agencyData["Offers Medical Social Services"]}
                    </div>
                  </li>
                  <li className="list-item">
                    {"Offers Nursing Care Services"}
                    <div className="services-item">
                    {agencyData["Offers Nursing Care Services"] === "Yes" ? <FaCheck color="green"/>   : <FaRegWindowClose color="red"/>}
                    {agencyData["Offers Nursing Care Services"]}
                    </div>
                  </li>
                  <li className="list-item">
                    {"Offers Occupational Therapy Services"}
                    <div className="services-item">
                    {agencyData["Offers Occupational Therapy Services"] === "Yes" ? <FaCheck color="green"/>   : <FaRegWindowClose color="red"/>}
                    {agencyData["Offers Occupational Therapy Services"]}
                    </div>
                  </li>
                  <li className="list-item">
                    {"Offers Physical Therapy Services"}
                    <div className="services-item">
                    {agencyData["Offers Physical Therapy Services"] === "Yes" ? <FaCheck color="green"/>   : <FaRegWindowClose color="red"/>}
                    {agencyData["Offers Physical Therapy Services"]}
                    </div>
                  </li>
                  <li className="list-item">
                    {"Offers Speech Pathology Services"}
                    <div className="services-item">
                    {agencyData["Offers Speech Pathology Services"] === "Yes" ? <FaCheck color="green"/>   : <FaRegWindowClose color="red"/>}
                    {agencyData["Offers Speech Pathology Services"]}
                    </div>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
      <ToastContainer
        position="bottom-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="dark"
      />
    </div>
  );
};

export default AgencySearch;
