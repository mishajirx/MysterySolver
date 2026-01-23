
import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface NetworkGraphProps {
    weights: any;
}

export function NetworkGraph({ weights }: NetworkGraphProps) {
    const svgRef = useRef<SVGSVGElement>(null);

    useEffect(() => {
        if (!weights || !svgRef.current) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll("*").remove();

        const width = 800;
        const height = 600;

        // Define layers: Input (subset), Hidden1 (subset), Hidden2 (subset), Output
        // Weights struct: layer1 (In->H1), layer2 (H1->H2), output (H2->Out)

        // We need to define node positions.
        // Let's visualize a subset of nodes since real dimensions (205, 512) are too big.
        // Backend already sends a subset if configured, or full. 
        // We'll trust the weights object structure: 
        // weights.layer1.weights is [Out, In] (PyTorch convention) -> [20, 20] subset

        const w1 = weights.layer1.weights; // [H1_sub, In_sub]
        const w2 = weights.layer2.weights; // [H2_sub, H1_sub]
        const wo = weights.output.weights; // [Out, H2_sub]

        const nodes: any[] = [];
        const links: any[] = [];

        // Config
        const layerGap = 200;
        const nodeGap = 25; // Adjusted for subset

        // 1. Input Layer
        const numInputs = w1[0].length; // Columns of w1
        const inputX = 100;
        const inputNodes = [];
        for (let i = 0; i < numInputs; i++) {
            const node = { id: `in-${i}`, x: inputX, y: (height - numInputs * nodeGap) / 2 + i * nodeGap, layer: 0 };
            nodes.push(node);
            inputNodes.push(node);
        }

        // 2. Hidden Layer 1
        const numH1 = w1.length; // Rows of w1
        const h1X = inputX + layerGap;
        const h1Nodes = [];
        for (let i = 0; i < numH1; i++) {
            const node = { id: `h1-${i}`, x: h1X, y: (height - numH1 * nodeGap) / 2 + i * nodeGap, layer: 1 };
            nodes.push(node);
            h1Nodes.push(node);
        }

        // Links In -> H1
        for (let i = 0; i < numInputs; i++) {
            for (let j = 0; j < numH1; j++) {
                const weightVal = w1[j][i];
                links.push({
                    source: inputNodes[i],
                    target: h1Nodes[j],
                    weight: weightVal
                });
            }
        }

        // 3. Hidden Layer 2
        const numH2 = w2.length;
        const h2X = h1X + layerGap;
        const h2Nodes = [];
        for (let i = 0; i < numH2; i++) {
            const node = { id: `h2-${i}`, x: h2X, y: (height - numH2 * nodeGap) / 2 + i * nodeGap, layer: 2 };
            nodes.push(node);
            h2Nodes.push(node);
        }

        // Links H1 -> H2
        for (let i = 0; i < numH1; i++) {
            for (let j = 0; j < numH2; j++) {
                const weightVal = w2[j][i];
                links.push({
                    source: h1Nodes[i],
                    target: h2Nodes[j],
                    weight: weightVal
                });
            }
        }

        // 4. Output Layer
        const numOut = wo.length;
        const outX = h2X + layerGap;
        const outNodes = [];
        for (let i = 0; i < numOut; i++) {
            const node = { id: `out-${i}`, x: outX, y: (height - numOut * 50) / 2 + i * 50, layer: 3 }; // More spacing for output
            nodes.push(node);
            outNodes.push(node);
        }

        // Links H2 -> Out
        for (let i = 0; i < numH2; i++) {
            for (let j = 0; j < numOut; j++) {
                const weightVal = wo[j][i];
                links.push({
                    source: h2Nodes[i],
                    target: outNodes[j],
                    weight: weightVal
                });
            }
        }

        // Drawing
        const g = svg.append("g");

        // Draw Links
        g.selectAll("line")
            .data(links)
            .enter()
            .append("line")
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y)
            .attr("stroke", d => d.weight > 0 ? "rgba(255, 100, 100, 0.5)" : "rgba(100, 100, 255, 0.5)")
            .attr("stroke-width", d => Math.min(Math.abs(d.weight) * 2, 3));

        // Draw Nodes
        const nodeGroup = g.selectAll(".node")
            .data(nodes)
            .enter()
            .append("g")
            .attr("class", "node")
            .attr("transform", d => `translate(${d.x},${d.y})`);

        nodeGroup.append("circle")
            .attr("r", 15)
            .attr("fill", "#1f2937")
            .attr("stroke", "#60a5fa")
            .attr("stroke-width", 2);

        nodeGroup.append("text")
            .attr("dy", ".35em")
            .attr("text-anchor", "middle")
            .text(d => {
                if (d.layer === 0) return "In";
                let val = 0;
                if (d.layer === 1) {
                    const idx = parseInt(d.id.split('-')[1]);
                    val = weights.layer1.bias[idx];
                } else if (d.layer === 2) {
                    const idx = parseInt(d.id.split('-')[1]);
                    val = weights.layer2.bias[idx];
                } else if (d.layer === 3) {
                    const idx = parseInt(d.id.split('-')[1]);
                    val = weights.output.bias[idx];
                }
                return val !== undefined ? val.toFixed(2) : "";
            })
            .attr("font-size", "10px")
            .attr("fill", "#fff");

    }, [weights]);

    return (
        <div className="w-full h-full flex justify-center items-center bg-black/20 rounded-xl overflow-hidden glass-panel">
            <svg ref={svgRef} width="100%" height="100%" viewBox="0 0 800 600" className="w-full h-full"></svg>
        </div>
    );
}
