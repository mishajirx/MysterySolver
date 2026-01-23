import * as tf from '@tensorflow/tfjs';
import Papa from 'papaparse';

export interface DataPoint {
    features: number[];
    label: number;
    id: number;
}

export class MysteryModel {
    model: tf.Sequential;
    inputDim = 205;

    constructor(
        learningRate: number = 0.001
    ) {
        this.model = tf.sequential();

        // First Layer: Linear -> ReLU
        this.model.add(tf.layers.dense({
            units: 512,
            inputShape: [this.inputDim],
            activation: 'relu',
            kernelInitializer: 'glorotUniform'
        }));

        // Second Layer: Linear -> ReLU
        this.model.add(tf.layers.dense({
            units: 512,
            activation: 'relu',
            kernelInitializer: 'glorotUniform'
        }));

        // Output Layer: Linear (logits for 5 classes)
        this.model.add(tf.layers.dense({
            units: 5,
            activation: 'linear', // Using linear to match PyTorch CrossEntropyLoss logic (logits)
            kernelInitializer: 'glorotUniform'
        }));

        this.compileModel(learningRate);
    }

    compileModel(learningRate: number) {
        this.model.compile({
            optimizer: tf.train.sgd(learningRate),
            loss: tf.losses.softmaxCrossEntropy, // Handles logits
            metrics: ['accuracy']
        });
    }

    async loadCSV(_lossFn: (progress: number) => void): Promise<{ xs: tf.Tensor2D, ys: tf.Tensor }> {
        return new Promise((resolve, reject) => {
            Papa.parse('/data/train.csv', {
                download: true,
                header: false,
                dynamicTyping: true,
                skipEmptyLines: true,
                worker: true, // Run in worker to avoid freezing UI
                step: (_row) => {
                    // We might implement streaming here if the file is too big, 
                    // but for 30MB we can likely load it all or chunk it.
                    // For simplicity in this v1, let's load all data first.
                },
                complete: (results) => {
                    const data = results.data as number[][];

                    // Assuming train.csv has NO header based on previous `head` command output
                    // Format: id, feature1...feature205, label

                    const features: number[][] = [];
                    const labels: number[] = [];

                    const totalRows = data.length;

                    for (let i = 0; i < totalRows; i++) {
                        const row = data[i];
                        // Verify row length (should be 1 + 205 + 1 = 207)
                        if (row.length < 207) continue;

                        // Skip ID (index 0)
                        // Features: index 1 to 205 (inclusive of start, exclusive of label) -> 1 to 206
                        const rowFeatures = row.slice(1, 206);
                        const rowLabel = row[206];

                        features.push(rowFeatures);
                        labels.push(rowLabel);
                    }

                    const xs = tf.tensor2d(features);
                    // TF.js wants labels as indices for sparse categorical crossentropy or one-hot for others.
                    // Let's use sparse logic or one-hot. Here let's just keep 1D
                    // Actually, softmaxCrossEntropy in tfjs usually expects one-hot encoding for the labels.
                    // Let's convert labels to one-hot.
                    const ysIndex = tf.tensor1d(labels, 'int32');
                    const ys = tf.oneHot(ysIndex, 5);

                    resolve({ xs, ys });
                },
                error: (err) => {
                    reject(err);
                }
            });
        });
    }

    // Helper to fetch the raw data directly for visualization
    static async parseDataUrl(url: string): Promise<DataPoint[]> {
        return new Promise((resolve, reject) => {
            Papa.parse(url, {
                download: true,
                header: false,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    const data = results.data as number[][];
                    const parsed: DataPoint[] = [];
                    for (let row of data) {
                        if (row.length < 207) continue;
                        parsed.push({
                            id: row[0],
                            features: row.slice(1, 206),
                            label: row[206]
                        })
                    }
                    resolve(parsed);
                },
                error: reject
            })
        })
    }

    getWeights() {
        // Returns weights for visualization
        // Layer 0: Dense 1
        // Layer 1: Dense 2
        // Layer 2: Output
        const layers = this.model.layers;
        return layers.map(l => {
            const w = l.getWeights();
            return {
                name: l.name,
                weights: w[0], // Kernel
                bias: w[1]    // Bias
            }
        });
    }
}
