
interface HyperparametersProps {
    epochs: number;
    setEpochs: (v: number) => void;
    learningRate: number;
    setLearningRate: (v: number) => void;
    batchSize: number;
    setBatchSize: (v: number) => void;
    onTrain: () => void;
    isTraining: boolean;
    status: 'idle' | 'loading_data' | 'training';
    onReset: () => void;
}

export function Hyperparameters({
    epochs, setEpochs,
    learningRate, setLearningRate,
    batchSize, setBatchSize,
    onTrain, isTraining, status, onReset
}: HyperparametersProps) {
    return (
        <div className="p-6 bg-white/5 backdrop-blur-md rounded-xl border border-white/10 shadow-xl text-gray-100">
            <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                <span className="w-1 h-6 bg-blue-500 rounded-full"></span>
                Training Controls
            </h2>

            <div className="space-y-5">
                <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Epochs</label>
                    <input
                        type="number"
                        value={epochs}
                        onChange={(e) => setEpochs(Number(e.target.value))}
                        disabled={isTraining}
                        className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                    />
                </div>

                <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Learning Rate</label>
                    <input
                        type="number" step="0.0001"
                        value={learningRate}
                        onChange={(e) => setLearningRate(Number(e.target.value))}
                        disabled={isTraining}
                        className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                    />
                </div>

                <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Batch Size</label>
                    <div className="relative">
                        <select
                            value={batchSize}
                            onChange={(e) => setBatchSize(Number(e.target.value))}
                            disabled={isTraining}
                            className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 appearance-none transition-all cursor-pointer"
                        >
                            {[32, 64, 128, 256].map(s => (
                                <option key={s} value={s} className="bg-gray-800">{s}</option>
                            ))}
                        </select>
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-gray-400">
                            ▼
                        </div>
                    </div>
                </div>

                <div className="flex space-x-3 pt-4">
                    <button
                        onClick={onTrain}
                        disabled={isTraining}
                        className={`flex-1 py-3 rounded-lg font-bold shadow-lg transition-all duration-300 transform active:scale-95 ${isTraining
                                ? 'bg-gray-700/50 cursor-not-allowed text-gray-400'
                                : 'bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white hover:shadow-blue-500/20'
                            }`}
                    >
                        {status === 'loading_data' ? (
                            <span className="flex items-center justify-center gap-2">
                                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Loading...
                            </span>
                        ) : status === 'training' ? (
                            'Training...'
                        ) : (
                            'Start Training'
                        )}
                    </button>
                    <button
                        onClick={onReset}
                        disabled={isTraining}
                        className="flex-1 py-3 rounded-lg font-bold bg-white/5 hover:bg-white/10 text-red-400 hover:text-red-300 border border-white/5 hover:border-red-500/30 transition-all duration-300 disabled:opacity-50"
                    >
                        Reset
                    </button>
                </div>
            </div>
        </div>
    );
}
