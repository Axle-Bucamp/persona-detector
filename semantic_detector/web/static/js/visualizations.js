// Visualization functions for 3D plots, histograms, and sentiment charts

// Add visualizations (3D graph, histograms, sentiment)
function addVisualizations(data) {
    const vizSection = document.getElementById('visualizations-section');
    if (!vizSection) return;
    
    let vizHtml = '<h3 style="margin-top: 2rem; margin-bottom: 1rem;">Visualizations</h3>';
    
    // 3D UMAP Projection
    if (data.coords_3d && data.coords_3d.length > 0 && data.labels) {
        vizHtml += '<div id="plot3d" style="width:100%; height:600px; margin-bottom: 2rem;"></div>';
    }
    
    // Sentiment Analysis
    if (data.overall_sentiment) {
        vizHtml += '<div id="sentiment-chart" style="width:100%; height:400px; margin-bottom: 2rem;"></div>';
    }
    
    // Word Distribution Histogram
    if (data.tfidf_features && Object.keys(data.tfidf_features).length > 0) {
        vizHtml += '<div id="word-distribution" style="width:100%; height:500px; margin-bottom: 2rem;"></div>';
    }
    
    vizSection.innerHTML = vizHtml;
    
    // Render 3D plot
    if (data.coords_3d && data.coords_3d.length > 0 && data.labels && typeof Plotly !== 'undefined') {
        setTimeout(() => render3DPlot(data.coords_3d, data.labels, data.sentences || []), 100);
    }
    
    // Render sentiment chart
    if (data.overall_sentiment && typeof Plotly !== 'undefined') {
        setTimeout(() => renderSentimentChart(data.overall_sentiment, data.cluster_sentiments || {}), 200);
    }
    
    // Render word distribution
    if (data.tfidf_features && Object.keys(data.tfidf_features).length > 0 && typeof Plotly !== 'undefined') {
        setTimeout(() => renderWordDistribution(data.tfidf_features), 300);
    }
}

// Render 3D UMAP plot
function render3DPlot(coords3d, labels, sentences) {
    try {
        const uniqueLabels = [...new Set(labels)];
        const colors = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6'];
        
        const traces = uniqueLabels.map((label, idx) => {
            const indices = coords3d.map((_, i) => i).filter(i => labels[i] === label);
            const x = indices.map(i => coords3d[i][0]);
            const y = indices.map(i => coords3d[i][1]);
            const z = indices.map(i => coords3d[i][2]);
            const texts = indices.map(i => sentences[i] ? sentences[i].substring(0, 100) : `Sentence ${i}`);
            
            return {
                x, y, z,
                mode: 'markers',
                type: 'scatter3d',
                name: `Cluster ${label}`,
                marker: {
                    size: 5,
                    color: colors[idx % colors.length],
                    opacity: 0.7
                },
                text: texts,
                hovertemplate: '<b>%{text}</b><br>Cluster: ' + label + '<extra></extra>'
            };
        });
        
        const layout = {
            title: '3D UMAP Projection',
            scene: {
                xaxis: { title: 'UMAP 1' },
                yaxis: { title: 'UMAP 2' },
                zaxis: { title: 'UMAP 3' }
            },
            height: 600
        };
        
        Plotly.newPlot('plot3d', traces, layout);
    } catch (e) {
        console.error('Error rendering 3D plot:', e);
    }
}

// Render sentiment chart
function renderSentimentChart(overallSentiment, clusterSentiments) {
    try {
        const labels = ['Positive', 'Neutral', 'Negative'];
        const overallValues = [
            overallSentiment.positive_percentage || 0,
            overallSentiment.neutral_percentage || 0,
            overallSentiment.negative_percentage || 0
        ];
        
        const traces = [{
            x: labels,
            y: overallValues,
            type: 'bar',
            marker: {
                color: ['#10b981', '#64748b', '#ef4444']
            },
            name: 'Overall Sentiment'
        }];
        
        // Add cluster sentiments if available
        Object.entries(clusterSentiments).forEach(([clusterId, sentiment]) => {
            if (sentiment && sentiment.average) {
                traces.push({
                    x: labels,
                    y: [
                        sentiment.average.positive_percentage || 0,
                        sentiment.average.neutral_percentage || 0,
                        sentiment.average.negative_percentage || 0
                    ],
                    type: 'bar',
                    name: `Cluster ${clusterId}`,
                    opacity: 0.7
                });
            }
        });
        
        const layout = {
            title: 'Sentiment Distribution',
            xaxis: { title: 'Sentiment' },
            yaxis: { title: 'Percentage (%)' },
            barmode: 'group',
            height: 400
        };
        
        Plotly.newPlot('sentiment-chart', traces, layout);
    } catch (e) {
        console.error('Error rendering sentiment chart:', e);
    }
}

// Render word distribution histogram
function renderWordDistribution(tfidfFeatures) {
    try {
        const traces = [];
        const colors = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'];
        
        Object.entries(tfidfFeatures).forEach(([clusterId, features], idx) => {
            if (features && features.length > 0) {
                const topFeatures = features.slice(0, 15);
                const words = topFeatures.map(f => Array.isArray(f) ? f[0] : (f.feature || f[0] || ''));
                const scores = topFeatures.map(f => Array.isArray(f) ? f[1] : (f.score || f[1] || 0));
                
                traces.push({
                    x: words,
                    y: scores,
                    type: 'bar',
                    name: `Cluster ${clusterId}`,
                    marker: { color: colors[idx % colors.length] }
                });
            }
        });
        
        if (traces.length > 0) {
            const layout = {
                title: 'Top TF-IDF Features by Cluster',
                xaxis: { title: 'Words/Phrases', tickangle: -45 },
                yaxis: { title: 'TF-IDF Score' },
                barmode: 'group',
                height: 500
            };
            
            Plotly.newPlot('word-distribution', traces, layout);
        }
    } catch (e) {
        console.error('Error rendering word distribution:', e);
    }
}

