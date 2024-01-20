import { useEffect, useState } from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
const AgencySearch = () => {
  const [agencyName, setAgencyName] = useState("");
  const [zipCode, setZipCode] = useState("");
  const [agencies, setAgencies] = useState([]);

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
        console.log(data);
        if (data.error) {
          toast.error(data.error);
          return;
        }
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
      {/* <label>
        Agency Name: <input name="myInput" onChange={handleNameChange} />
      </label>
      <button>Submit</button>
      <p>{agencyName}</p> */}
      <div className="search-page">
        <div className="search-container">
          <div className="search-inner">
            <input type="text" value={agencyName} onChange={handleNameChange} />
            <input type="text" value={zipCode} onChange={handleZipCodeChange} />
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
                  agency !== searchTerm
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
