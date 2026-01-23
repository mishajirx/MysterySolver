import { useEffect, useRef, useState } from 'react'; import * as tf from '@tensorflow/tfjs';

interface WeightVisualizerProps {
    weights: { name: string, weights: tf.Tensor, bias: tf.Tensor }[];
}

export function WeightVisualizer({ weights }: WeightVisualizerProps) {
    const [selectedLayer, setSelectedLayer] = useState(0);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [matrixData, setMatrixData] = useState<number[][] | null>(null);

    useEffect(() => {
        if (!weights || weights.length === 0) return;

        // Process selected layer weights to array
        const process = async () => {
            const tensor = weights[selectedLayer].weights;
            const array = await tensor.array() as number[][];
            setMatrixData(array);
        }
        process();

    }, [weights, selectedLayer]);

    useEffect(() => {
        // Draw heatmap
        if (!matrixData || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const rows = matrixData.length;
        const cols = matrixData[0].length;

        // Limit canvas size for performance. If too big, maybe simple sampling or fixed px.
        // Heatmap logic: map value to color.
        // Normalize values roughly between -0.5 and 0.5 usually?

        // Auto-scale
        const pixelSize = Math.max(1, Math.floor(600 / Math.max(rows, cols)));
        canvas.width = cols * pixelSize;
        canvas.height = rows * pixelSize;

        // Find max abs for normalization
        let maxVal = 0;
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                maxVal = Math.max(maxVal, Math.abs(matrixData[r][c]));
            }
        }

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const val = matrixData[r][c];
                const norm = val / (maxVal || 1); // -1 to 1

                // Color map: Blue (-) -> White (0) -> Red (+)
                let rVal, gVal, bVal;
                if (norm < 0) {
                    // negative -> blue
                    // 0 -> white(255,255,255)
                    // -1 -> blue(0,0,255)
                    const intensity = Math.floor(255 * (1 + norm)); // norm is -0.5 -> 0.5 * 255
                    rVal = intensity;
                    gVal = intensity;
                    bVal = 255;
                } else {
                    // positive -> red
                    const intensity = Math.floor(255 * (1 - norm));
                    rVal = 255;
                    gVal = intensity;
                    bVal = intensity;
                }

                ctx.fillStyle = `rgb(${rVal},${gVal},${bVal})`;
                ctx.fillRect(c * pixelSize, r * pixelSize, pixelSize, pixelSize);
            }
        }

    }, [matrixData]);

    return (
        <div className="p-6 bg-white/5 backdrop-blur-md rounded-xl border border-white/10 shadow-xl h-full flex flex-col">
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-bold text-gray-100 flex items-center gap-2">
                    <span className="w-1 h-6 bg-pink-500 rounded-full"></span>
                    Weight Heatmap
                </h2>
                <div className="relative">
                    <select
                        className="appearance-none bg-black/20 border border-white/10 text-white rounded-lg pl-3 pr-8 py-1 focus:outline-none focus:ring-2 focus:ring-pink-500/50 cursor-pointer"
                        value={selectedLayer}
                        onChange={(e) => setSelectedLayer(Number(e.target.value))}
                    >
                        {weights.map((w, i) => (
                            <option key={i} value={i} className="bg-gray-800">{w.name}</option>
                        ))}
                    </select>
                    <div className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-gray-400 text-xs">
                        ▼
                    </div>
                </div>
            </div>

            <div className="flex-1 overflow-auto bg-black/40 rounded-lg border border-white/5 flex justify-center items-center backdrop-blur-sm p-4">
                {!matrixData ? (
                    <div className="flex flex-col items-center gap-3 text-gray-400">
                        <svg className="animate-spin h-8 w-8 opacity-50" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>Loading weights...</span>
                    </div>
                ) : (
                    <canvas ref={canvasRef} className="shadow-2xl rounded-sm max-w-full max-h-full object-contain pixelated" style={{ imageRendering: 'pixelated' }} />
                )}
            </div>
            <div className="text-gray-400 text-xs mt-3 text-center font-mono bg-black/20 py-2 rounded border border-white/5">
                Rows: Input Dim | Cols: Output Dim <span className="mx-2 text-gray-600">|</span>
                <span className="text-blue-400">Blue: -</span> <span className="text-white">White: 0</span> <span className="text-red-400">Red: +</span>
            </div>
        </div>
    );

}
