import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

interface TrainingMonitorProps {
    logs: { epoch: number, loss: number, acc: number }[];
}

export function TrainingMonitor({ logs }: TrainingMonitorProps) {
    const data = {
        labels: logs.map(l => l.epoch),
        datasets: [
            {
                label: 'Loss',
                data: logs.map(l => l.loss),
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                yAxisID: 'y',
            },
            {
                label: 'Accuracy',
                data: logs.map(l => l.acc),
                borderColor: 'rgb(53, 162, 235)',
                backgroundColor: 'rgba(53, 162, 235, 0.5)',
                yAxisID: 'y1',
            },
        ],
    };

    const options = {
        responsive: true,
        interaction: {
            mode: 'index' as const,
            intersect: false,
        },
        stacked: false,
        scales: {
            x: {
                grid: { color: "#444" },
                ticks: { color: "#ddd" }
            },
            y: {
                type: 'linear' as const,
                display: true,
                position: 'left' as const,
                grid: { color: "#444" },
                ticks: { color: "#ddd" }
            },
            y1: {
                type: 'linear' as const,
                display: true,
                position: 'right' as const,
                grid: {
                    drawOnChartArea: false,
                },
                ticks: { color: "#ddd" }
            },
        },
        plugins: {
            legend: {
                labels: { color: "#fff" }
            }
        }
    };

    return (
        <div className="p-6 bg-white/5 backdrop-blur-md rounded-xl border border-white/10 shadow-xl h-full flex flex-col">
            <h2 className="text-xl font-bold mb-4 text-gray-100 flex items-center gap-2">
                <span className="w-1 h-6 bg-purple-500 rounded-full"></span>
                Training Metrics
            </h2>
            {logs.length === 0 ? (
                <div className="flex-1 flex flex-col items-center justify-center text-gray-500 gap-3 border-2 border-dashed border-gray-700/50 rounded-lg m-2">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <span>No training data yet</span>
                </div>
            ) : (
                <div className="flex-1 relative w-full h-full min-h-0">
                    <Line options={{ ...options, maintainAspectRatio: false }} data={data} />
                </div>
            )}
        </div>
    );
}
