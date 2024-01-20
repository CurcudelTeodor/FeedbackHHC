import { useEffect, useState } from "react";
import { BarChart } from "./BarChart";
import Select from "react-select";

const Chart = () => {
  const [dataForChart, setDataForChart] = useState();
  const [optionsData, setOptionsData] = useState([]);
  const [histogramData, setHistogramData] = useState();

  const handleSelectChange = (selectedOption) => {
    const dataForChart = transformHistogramDataToChartData(
      histogramData[selectedOption.value]
    );
    setDataForChart(dataForChart);
  };
  const fetchColumns = () => {
    fetch("http://127.0.0.1:5000")
      .then((res) => res.json())
      .then((d) => {
        for (let i = 0; i < d.partitions.length; i++) {
          optionsData.push({ value: d.partitions[i], label: d.partitions[i] });
        }
      });
    setOptionsData(optionsData);
  };

  const fetchData = () => {
    fetch("http://127.0.0.1:5000/histograms")
      .then((res) => res.json())
      .then((data) => {
        setHistogramData(data);

        setDataForChart(dataForChart);
      });
  };

  useEffect(() => {
    fetchColumns();
    fetchData();
  }, []);

  const transformHistogramDataToChartData = (histogramData) => {
    return {
      labels: histogramData.bins,
      datasets: [
        {
          label: "Frequency",
          data: histogramData.count,
          backgroundColor: "rgba(0, 123, 255, 0.6)",
          borderColor: "rgba(0, 123, 255, 0.6)",
          borderWidth: 0,
        },
      ],
    };
  };
  return (
    <div className="container">
          <div className="selector">
            <Select
              defaultValue={optionsData[0]}
              options={optionsData}
              onChange={handleSelectChange}
              theme={(theme) => ({
                ...theme,
                borderRadius: 0,
                colors: {
                  ...theme.colors,
                  primary25: "white",
                  primary: "gray",
                },
              })}
              styles={{
                option: (provided) => ({
                  ...provided,
                  color: "black",
                }),
                singleValue: (provided) => ({
                  ...provided,
                  color: "black",
                }),
                control: (provided) => ({
                  ...provided,
                  height: "50px",
                  width: "300px",
                }),
                menu: (provided) => ({
                  ...provided,
                  width: "300px",
                }),
              }}
            />
          </div>
          <div className="chart-container">
            {dataForChart && <BarChart data={dataForChart} />}
          </div>
        </div>
  )
}

export default Chart;