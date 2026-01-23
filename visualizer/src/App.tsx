
import { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { MysteryModel } from './lib/model';
import { Hyperparameters } from './components/Hyperparameters';
import { TrainingMonitor } from './components/TrainingMonitor';
import { WeightVisualizer } from './components/WeightVisualizer';

function App() {
  const [model, setModel] = useState<MysteryModel | null>(null);
  const [epochs, setEpochs] = useState(30);
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchSize, setBatchSize] = useState(64);
  const [status, setStatus] = useState<'idle' | 'loading_data' | 'training'>('idle');
  const [isTraining, setIsTraining] = useState(false);

  const [logs, setLogs] = useState<{ epoch: number, loss: number, acc: number }[]>([]);
  const [currentWeights, setCurrentWeights] = useState<{ name: string, weights: tf.Tensor, bias: tf.Tensor }[]>([]);

  // Keep tf tensors references to dispose if needed, though for visualization 
  // we iterate and the visualizer will likely consume copies or arrays.
  const initModel = async () => {
    if (model) {
      model.model.dispose();
    }

    const newModel = new MysteryModel(learningRate);
    setModel(newModel);
    setLogs([]);
    setCurrentWeights(newModel.getWeights());
  };

  useEffect(() => {
    initModel();
    return () => { }
  }, []);

  const handleReset = () => {
    initModel();
    setStatus('idle');
  }

  const handleTrain = async () => {
    if (!model) return;
    setStatus('loading_data');
    setIsTraining(true);

    model.compileModel(learningRate);

    try {
      // Small timeout to allow UI to update to "Loading Data..."
      await new Promise(r => setTimeout(r, 100));

      const { xs, ys } = await model.loadCSV((p) => console.log('Parsed', p));

      setStatus('training');

      await model.model.fit(xs, ys, {
        epochs: epochs,
        batchSize: batchSize,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: (epoch, log) => {
            if (log) {
              setLogs(prev => [...prev, {
                epoch: epoch + 1,
                loss: log.loss,
                acc: log.acc
              }]);
              setCurrentWeights(model.getWeights());
            }
            return tf.nextFrame();
          }
        }
      });

      xs.dispose();
      ys.dispose();
    } catch (err) {
      console.error("Training failed", err);
      alert("Training error: " + err);
    } finally {
      setIsTraining(false);
      setStatus('idle');
    }
  };

  return (
    <div className="min-h-screen text-gray-100 p-8 font-sans">
      <header className="mb-8 flex justify-between items-center bg-white/5 backdrop-blur-md p-4 rounded-xl border border-white/10 shadow-xl">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-500/20 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
            Mystery Solver <span className="text-gray-400 font-medium">Visualizer</span>
          </h1>
        </div>
        <div className="text-xs font-mono px-3 py-1 bg-black/30 rounded-full text-blue-300 border border-blue-500/30">
          TF.js Backend: {tf.getBackend()}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-10rem)]">
        {/* Left Sidebar: Controls & Metrics */}
        <div className="lg:col-span-1 flex flex-col space-y-6 overflow-y-auto pr-2">
          <Hyperparameters
            epochs={epochs} setEpochs={setEpochs}
            learningRate={learningRate} setLearningRate={setLearningRate}
            batchSize={batchSize} setBatchSize={setBatchSize}
            onTrain={handleTrain}
            isTraining={isTraining}
            status={status}
            onReset={handleReset}
          />

          <div className="flex-1 min-h-[300px]">
            <TrainingMonitor logs={logs} />
          </div>
        </div>

        {/* Main Area: Visualizations */}
        <div className="lg:col-span-3 h-full">
          <WeightVisualizer weights={currentWeights} />
        </div>
      </div>
    </div>
  )
}

export default App
