import { useEffect, useState } from "react";
import { ToastContainer, toast } from "react-toastify";
import { FaCheck,FaRegWindowClose  } from "react-icons/fa";
import "react-toastify/dist/ReactToastify.css";
import { SERVER } from "../config";


const Predict = () => {
    const [instance, setInstance] = useState('');
    const [predictions, setPredictions] = useState([]);

    const handlePredict = async () => {
        const options = {
            method: 'POST',
            headers: {
                'content-type': 'application/json',
                'access-control-allow-origin': 'no-cors'
            }, 
            body: JSON.stringify({'instance': instance})
        }

        try {
            const res = await fetch(`${SERVER}/predict`, options);
            if(res.status >= 400) {
                const data = await res.json();
                throw data.error;
            }
    
            const data = await res.json();

            const info = data.info;
            
            console.log(info);

            const pred = [];
            for(const key in info) {
                pred.push([key, info[key]]);
            }

            setPredictions(pred);
        } catch(error) {
            toast.error(error);
        }
    };

    return (
        <div>
            <div className="search-page">
                <div className="search-container">
                    <div className="search-inner">
                        <input
                            type="text"
                              value={instance}
                              onChange={e => setInstance(e.target.value)}
                            placeholder="Input your instance as a csv line"
                        />
                        <button className="search-btn" onClick={handlePredict}> Go! </button>
                    </div>
                </div>
            </div>

            <div style={{padding: 20}}></div>

            <table>
                <thead>
                    <tr>
                    <th>Classifier</th>
                    <th>Prediction</th>
                    </tr>
                </thead>
                <tbody>
                    {predictions.map(([classifier, prediction]) => {
                        return (
                        <tr key={classifier}>
                            <td>{classifier}</td>
                            <td>{prediction}</td>
                        </tr>
                    )})}
                </tbody>
            </table>

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

export default Predict;
