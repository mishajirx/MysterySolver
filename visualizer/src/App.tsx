import { useEffect, useState } from 'react';
import axios from 'axios';
import { Hyperparameters } from './components/Hyperparameters';
import { NetworkGraph } from './components/NetworkGraph';

interface TrainingLog {
  loss: number;
  accuracy: number;
}

function App() {
  const [epochs, setEpochs] = useState(1); // Epochs per step
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchSize, setBatchSize] = useState(64); // Passed but handled on backend usually
  const [status, setStatus] = useState<'idle' | 'training'>('idle');

  const [logs, setLogs] = useState<TrainingLog | null>(null);
  const [networkWeights, setNetworkWeights] = useState<any>(null);

  const API_URL = 'http://localhost:8000';

  const fetchWeights = async () => {
    try {
      const res = await axios.get(`${API_URL}/model/weights`);
      setNetworkWeights(res.data);
    } catch (err) {
      console.error("Failed to fetch weights", err);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchWeights();
  }, []);

  const handleReset = async () => {
    try {
      await axios.post(`${API_URL}/reset`);
      setStatus('idle');
      setLogs(null);
      fetchWeights();
    } catch (err) {
      console.error("Reset failed", err);
    }
  }

  const handleTrain = async () => {
    setStatus('training');
    try {
      // Run a training step
      const res = await axios.post(`${API_URL}/train/step`, null, {
        params: { epochs: epochs, lr: learningRate }
      });

      setLogs(res.data);
      await fetchWeights();

    } catch (err) {
      console.error("Training failed", err);
      // alert("Training error. Is backend running?");
    } finally {
      setStatus('idle');
    }
  };

  return (
    <div className="min-h-screen text-gray-100 p-8 font-sans selection:bg-pink-500/30">
      <header className="mb-8 flex justify-between items-center bg-white/5 backdrop-blur-md p-4 rounded-xl border border-white/10 shadow-xl">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-tr from-purple-500/20 to-blue-500/20 rounded-lg border border-white/5">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-purple-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="2" y1="12" x2="22" y2="12"></line>
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
            </svg>
          </div>
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400">
            Mystery Solver <span className="text-gray-500 font-medium">Network</span>
          </h1>
        </div>

        <div className="flex items-center gap-4">
          {logs && (
            <div className="flex gap-4 text-sm font-mono bg-black/30 px-4 py-1 rounded-full border border-white/5">
              <span className="text-red-300">Loss: {logs.loss.toFixed(4)}</span>
              <span className="text-green-300">Acc: {(logs.accuracy * 100).toFixed(1)}%</span>
            </div>
          )}
          <div className="text-xs font-mono px-3 py-1 bg-green-900/30 rounded-full text-green-400 border border-green-500/30">
            PyTorch Backend
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-10rem)]">
        {/* Left Sidebar: Controls */}
        <div className="lg:col-span-1 flex flex-col space-y-6">
          <Hyperparameters
            epochs={epochs} setEpochs={setEpochs}
            learningRate={learningRate} setLearningRate={setLearningRate}
            batchSize={batchSize} setBatchSize={setBatchSize}
            onTrain={handleTrain}
            isTraining={status === 'training'}
            status={status === 'training' ? 'training' : 'idle'}
            onReset={handleReset}
          />

          <div className="p-6 bg-white/5 backdrop-blur-md rounded-xl border border-white/10 shadow-xl flex-1 text-sm text-gray-400">
            <h3 className="font-bold text-gray-200 mb-2">Network Structure</h3>
            <ul className="space-y-1 font-mono">
              <li>Input: 205 features</li>
              <li>Hidden 1: 512 Neurons (ReLU)</li>
              <li>Hidden 2: 512 Neurons (ReLU)</li>
              <li>Output: 5 Classes</li>
            </ul>
            <p className="mt-4 text-xs italic opacity-50">
              Displaying a subset of neurons for clarity.
            </p>
          </div>
        </div>

        {/* Main Area: Network Graph */}
        <div className="lg:col-span-3 h-full bg-white/5 backdrop-blur-md rounded-xl border border-white/10 shadow-xl p-1 relative overflow-hidden">
          {networkWeights ? (
            <NetworkGraph weights={networkWeights} />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-gray-400 animate-pulse">
              Connecting to Backend...
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
