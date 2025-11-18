// Tab switching
document.addEventListener('DOMContentLoaded', function() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Remove active class from all buttons and contents
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            btn.classList.add('active');
            document.getElementById(`${targetTab}-tab`).classList.add('active');
        });
    });

    // Analyze form submission
    const analyzeForm = document.getElementById('analyze-form');
    analyzeForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handleAnalyze();
    });

    // Upload form submission
    const uploadForm = document.getElementById('upload-form');
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handleUpload();
    });

    // Style search form submission
    const styleSearchForm = document.getElementById('style-search-form');
    styleSearchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handleStyleSearch();
    });
});

// Show/hide loading indicator
function setLoading(show) {
    const loading = document.getElementById('loading');
    if (show) {
        loading.classList.remove('hidden');
    } else {
        loading.classList.add('hidden');
    }
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
    setTimeout(() => {
        errorDiv.classList.add('hidden');
    }, 5000);
}

// Hide error message
function hideError() {
    document.getElementById('error-message').classList.add('hidden');
}

// Handle analyze request
async function handleAnalyze() {
    hideError();
    setLoading(true);
    
    const formData = {
        text: document.getElementById('text-input').value,
        ollama_endpoint: document.getElementById('ollama-endpoint').value || null,
        embedding_model: document.getElementById('embedding-model').value || null,
        n_clusters: parseInt(document.getElementById('n-clusters').value) || null,
        clustering_method: document.getElementById('clustering-method').value
    };

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Analysis failed');
        }

        displayResults(data);
    } catch (error) {
        showError(`Error: ${error.message}`);
        console.error('Analysis error:', error);
    } finally {
        setLoading(false);
    }
}

// Handle file upload
async function handleUpload() {
    hideError();
    setLoading(true);

    const fileInput = document.getElementById('file-input');
    if (!fileInput.files[0]) {
        showError('Please select a file');
        setLoading(false);
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('ollama_endpoint', document.getElementById('upload-ollama-endpoint').value || '');
    formData.append('embedding_model', document.getElementById('upload-embedding-model').value || '');
    
    const nClusters = document.getElementById('upload-n-clusters').value;
    if (nClusters) {
        formData.append('n_clusters', nClusters);
    }
    formData.append('clustering_method', document.getElementById('upload-clustering-method').value);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Upload failed');
        }

        displayResults(data);
    } catch (error) {
        showError(`Error: ${error.message}`);
        console.error('Upload error:', error);
    } finally {
        setLoading(false);
    }
}

// Handle style search
async function handleStyleSearch() {
    hideError();
    setLoading(true);

    const formData = {
        sentence_index: parseInt(document.getElementById('sentence-index').value),
        k: parseInt(document.getElementById('k-results').value),
        in_cluster_only: document.getElementById('in-cluster-only').checked,
        out_cluster_only: document.getElementById('out-cluster-only').checked
    };

    try {
        const response = await fetch('/api/style-search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Style search failed');
        }

        displayStyleResults(data);
    } catch (error) {
        showError(`Error: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

// Display analysis results
function displayResults(data) {
    const resultsSection = document.getElementById('results-section');
    const resultsContent = document.getElementById('results-content');
    
    let html = `
        <div class="results-grid">
            <div class="result-card">
                <h3><i class="fas fa-chart-bar"></i> Statistics</h3>
                <div class="stat">
                    <span class="stat-label">Sentences:</span>
                    <span class="stat-value">${data.num_sentences}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Clusters:</span>
                    <span class="stat-value">${data.num_clusters}</span>
                </div>
            </div>
        </div>
    `;

    // Cluster stats
    if (Object.keys(data.cluster_stats).length > 0) {
        html += '<h3 style="margin-top: 2rem; margin-bottom: 1rem;">Cluster Statistics</h3>';
        html += '<div class="results-grid">';
        
        for (const [clusterId, stats] of Object.entries(data.cluster_stats)) {
            html += `
                <div class="result-card">
                    <h3>Cluster ${clusterId}</h3>
                    <div class="stat">
                        <span class="stat-label">Count:</span>
                        <span class="stat-value">${stats.count}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Avg Length:</span>
                        <span class="stat-value">${stats.avg_length.toFixed(1)} chars</span>
                    </div>
                </div>
            `;
        }
        html += '</div>';
    }

    // Fingerprint summaries
    if (Object.keys(data.fingerprint_summaries).length > 0) {
        html += '<h3 style="margin-top: 2rem; margin-bottom: 1rem;">Fingerprint Summaries</h3>';
        html += '<div class="results-grid">';
        
        for (const [clusterId, summary] of Object.entries(data.fingerprint_summaries)) {
            html += `
                <div class="result-card">
                    <h3>Cluster ${clusterId} Fingerprint</h3>
                    <div class="stat">
                        <span class="stat-label">Avg Sentence Length:</span>
                        <span class="stat-value">${summary.avg_sentence_length.toFixed(1)}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Vocab Richness:</span>
                        <span class="stat-value">${summary.vocab_richness.toFixed(3)}</span>
                    </div>
                    ${summary.top_words.length > 0 ? `
                        <div style="margin-top: 1rem;">
                            <strong>Top Words:</strong>
                            <ul class="fingerprint-list">
                                ${summary.top_words.slice(0, 5).map(w => `<li>${w}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            `;
        }
        html += '</div>';
    }

    // Sentiment Analysis
    if (data.overall_sentiment) {
        html += '<h3 style="margin-top: 2rem; margin-bottom: 1rem;">Sentiment Analysis</h3>';
        html += '<div class="results-grid">';
        html += `
            <div class="result-card">
                <h3>Overall Sentiment</h3>
                <div class="stat">
                    <span class="stat-label">Average Score:</span>
                    <span class="stat-value">${(data.overall_sentiment.average_sentiment_score * 100).toFixed(1)}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Person Score:</span>
                    <span class="stat-value">${data.overall_person_score.toFixed(1)}/100</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Positive:</span>
                    <span class="stat-value">${data.overall_sentiment.positive_percentage.toFixed(1)}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Neutral:</span>
                    <span class="stat-value">${data.overall_sentiment.neutral_percentage.toFixed(1)}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Negative:</span>
                    <span class="stat-value">${data.overall_sentiment.negative_percentage.toFixed(1)}%</span>
                </div>
            </div>
        `;
        
        // Cluster sentiments
        if (data.cluster_sentiments) {
            for (const [clusterId, sentiment] of Object.entries(data.cluster_sentiments)) {
                html += `
                    <div class="result-card">
                        <h3>Cluster ${clusterId} Sentiment</h3>
                        <div class="stat">
                            <span class="stat-label">Person Score:</span>
                            <span class="stat-value">${sentiment.person_score.toFixed(1)}/100</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Positive:</span>
                            <span class="stat-value">${sentiment.average.positive_percentage.toFixed(1)}%</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Neutral:</span>
                            <span class="stat-value">${sentiment.average.neutral_percentage.toFixed(1)}%</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Negative:</span>
                            <span class="stat-value">${sentiment.average.negative_percentage.toFixed(1)}%</span>
                        </div>
                    </div>
                `;
            }
        }
        html += '</div>';
    }
    
    // TF-IDF Features
    if (data.tfidf_features && Object.keys(data.tfidf_features).length > 0) {
        html += '<h3 style="margin-top: 2rem; margin-bottom: 1rem;">TF-IDF Style Features</h3>';
        html += '<div class="results-grid">';
        for (const [clusterId, features] of Object.entries(data.tfidf_features)) {
            if (features && features.length > 0) {
                html += `
                    <div class="result-card">
                        <h3>Cluster ${clusterId} Top Features</h3>
                        <ul class="fingerprint-list">
                            ${features.slice(0, 10).map(f => {
                                const word = Array.isArray(f) ? f[0] : f.feature || f[0];
                                const score = Array.isArray(f) ? f[1] : f.score || f[1];
                                return `<li>${word} (${score.toFixed(3)})</li>`;
                            }).join('')}
                        </ul>
                    </div>
                `;
            }
        }
        html += '</div>';
    }
    
    // Output files
    if (Object.keys(data.output_files).length > 0) {
        html += '<h3 style="margin-top: 2rem; margin-bottom: 1rem;">Generated Files</h3>';
        html += '<div class="results-grid">';
        for (const [key, file] of Object.entries(data.output_files)) {
            html += `
                <div class="result-card">
                    <h3>${key.toUpperCase()}</h3>
                    <a href="/${file}" target="_blank">
                        <i class="fas fa-external-link-alt"></i> View ${key}
                    </a>
                </div>
            `;
        }
        html += '</div>';
    }

    resultsContent.innerHTML = html;
    
    // Add visualizations if function exists
    if (typeof addVisualizations === 'function') {
        addVisualizations(data);
    }
    
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Display style search results
function displayStyleResults(data) {
    const resultsSection = document.getElementById('results-section');
    const resultsContent = document.getElementById('results-content');
    
    let html = `
        <div class="result-card">
            <h3>Query Sentence</h3>
            <p style="margin: 1rem 0; padding: 1rem; background: #eff6ff; border-radius: 0.5rem;">
                <strong>Index ${data.query_index}</strong> (Cluster ${data.query_cluster}):<br>
                "${data.query_text}"
            </p>
        </div>
    `;

    // Fingerprint summary
    if (data.fingerprint_summary) {
        html += `
            <div class="result-card">
                <h3>Fingerprint Summary</h3>
                <div class="stat">
                    <span class="stat-label">Avg Sentence Length:</span>
                    <span class="stat-value">${data.fingerprint_summary.avg_sentence_length.toFixed(1)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Vocab Richness:</span>
                    <span class="stat-value">${data.fingerprint_summary.vocab_richness.toFixed(3)}</span>
                </div>
                ${data.fingerprint_summary.top_words.length > 0 ? `
                    <div style="margin-top: 1rem;">
                        <strong>Top Words:</strong>
                        <ul class="fingerprint-list">
                            ${data.fingerprint_summary.top_words.slice(0, 5).map(w => `<li>${w}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        `;
    }

    // In-cluster matches
    if (data.in_cluster_matches && data.in_cluster_matches.length > 0) {
        html += '<h3 style="margin-top: 2rem;">In-Cluster Matches</h3>';
        data.in_cluster_matches.forEach(match => {
            html += `
                <div class="style-match">
                    <div class="style-match-header">
                        <span class="similarity-badge">${(match.similarity * 100).toFixed(1)}%</span>
                        <span class="cluster-badge">Cluster ${match.cluster_id}</span>
                    </div>
                    <p><strong>Index ${match.index}:</strong> "${match.text}"</p>
                </div>
            `;
        });
    }

    // Out-cluster matches
    if (data.out_cluster_matches && data.out_cluster_matches.length > 0) {
        html += '<h3 style="margin-top: 2rem;">Out-Cluster Matches</h3>';
        data.out_cluster_matches.forEach(match => {
            html += `
                <div class="style-match">
                    <div class="style-match-header">
                        <span class="similarity-badge">${(match.similarity * 100).toFixed(1)}%</span>
                        <span class="cluster-badge">Cluster ${match.cluster_id}</span>
                    </div>
                    <p><strong>Index ${match.index}:</strong> "${match.text}"</p>
                </div>
            `;
        });
    }

    resultsContent.innerHTML = html;
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

