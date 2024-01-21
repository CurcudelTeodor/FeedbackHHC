import {Bar} from "react-chartjs-2";
import { Chart as ChartJS} from 'chart.js/auto';


export const BarChart = ({data}) => {
  return (
    <div>
      <Bar width={600} height={600} data={data} />
    </div>
  )
}
