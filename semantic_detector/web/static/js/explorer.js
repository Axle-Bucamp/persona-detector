// Data Explorer functionality

let currentParseData = null;
let currentUnifiedData = null;
let currentTablePage = 1;
const rowsPerPage = 50;

// Initialize explorer tab
document.addEventListener('DOMContentLoaded', function() {
    const formatSelect = document.getElementById('explorer-format');
    const csvOptions = document.getElementById('csv-options');
    const jsonOptions = document.getElementById('json-options');
    const regexOptions = document.getElementById('regex-options');
    const markdownOptions = document.getElementById('markdown-options');
    const pdfOptions = document.getElementById('pdf-options');
    const pdfWarning = document.getElementById('pdf-warning');
    const parseBtn = document.getElementById('parse-btn');
    const analyzeBtn = document.getElementById('analyze-explorer-btn');
    const exportBtn = document.getElementById('export-csv-btn');
    const clusterFilter = document.getElementById('cluster-filter');
    const tableSearch = document.getElementById('table-search');
    
    // Format selection handler
    formatSelect.addEventListener('change', function() {
        csvOptions.classList.add('hidden');
        jsonOptions.classList.add('hidden');
        regexOptions.classList.add('hidden');
        markdownOptions.classList.add('hidden');
        pdfOptions.classList.add('hidden');
        pdfWarning.classList.add('hidden');
        
        if (formatSelect.value === 'csv') {
            csvOptions.classList.remove('hidden');
            // Load CSV columns when file is selected
            const fileInput = document.getElementById('explorer-file');
            if (fileInput.files.length > 0) {
                loadCSVColumns(fileInput.files[0]);
            }
        } else if (formatSelect.value === 'json') {
            jsonOptions.classList.remove('hidden');
        } else if (formatSelect.value === 'regex') {
            regexOptions.classList.remove('hidden');
        } else if (formatSelect.value === 'markdown') {
            markdownOptions.classList.remove('hidden');
        } else if (formatSelect.value === 'pdf') {
            pdfOptions.classList.remove('hidden');
            pdfWarning.classList.remove('hidden');
        }
    });
    
    // File input handler for CSV
    document.getElementById('explorer-file').addEventListener('change', function(e) {
        if (formatSelect.value === 'csv' && e.target.files.length > 0) {
            loadCSVColumns(e.target.files[0]);
        }
    });
    
    // Parse button handler
    parseBtn.addEventListener('click', handleParse);
    
    // Analyze button handler
    analyzeBtn.addEventListener('click', handleAnalyzeExplorer);
    
    // Export button handler
    exportBtn.addEventListener('click', handleExportCSV);
    
    // Filter handlers
    clusterFilter.addEventListener('change', filterTable);
    tableSearch.addEventListener('input', filterTable);
});

// Load CSV columns
async function loadCSVColumns(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/explorer/columns', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        const columnSelect = document.getElementById('csv-column');
        columnSelect.innerHTML = '<option value="">Select column...</option>';
        
        data.columns.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            columnSelect.appendChild(option);
        });
    } catch (error) {
        showError(`Error loading columns: ${error.message}`);
    }
}

// Handle parse
async function handleParse() {
    hideError();
    setLoading(true);
    
    const fileInput = document.getElementById('explorer-file');
    const formatSelect = document.getElementById('explorer-format');
    
    if (!fileInput.files[0]) {
        showError('Please select a file');
        setLoading(false);
        return;
    }
    
    if (!formatSelect.value) {
        showError('Please select a format type');
        setLoading(false);
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('format_type', formatSelect.value);
    
    // Add format-specific parameters
    if (formatSelect.value === 'csv') {
        const column = document.getElementById('csv-column').value;
        if (!column) {
            showError('Please select a text column');
            setLoading(false);
            return;
        }
        formData.append('text_column', column);
    } else if (formatSelect.value === 'json') {
        const jsonpath = document.getElementById('json-path').value;
        if (!jsonpath) {
            showError('Please enter a JSONPath expression');
            setLoading(false);
            return;
        }
        formData.append('jsonpath', jsonpath);
    } else if (formatSelect.value === 'regex') {
        const pattern = document.getElementById('regex-pattern').value;
        if (!pattern) {
            showError('Please enter a regex pattern');
            setLoading(false);
            return;
        }
        formData.append('regex_pattern', pattern);
        formData.append('use_capture_group', document.getElementById('use-capture-group').checked);
    } else if (formatSelect.value === 'markdown') {
        const mode = document.getElementById('markdown-mode').value;
        formData.append('markdown_mode', mode || 'sections');
    } else if (formatSelect.value === 'pdf') {
        const mode = document.getElementById('pdf-mode').value;
        formData.append('markdown_mode', mode || 'sections');
    }
    
    try {
        const response = await fetch('/api/explorer/parse', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Parse failed');
        }
        
        currentParseData = data;
        displayParsePreview(data);
        
        // Show PDF warning if applicable
        if (data.metadata && data.metadata.pdf_warning) {
            const pdfWarning = document.getElementById('pdf-warning');
            if (pdfWarning) {
                pdfWarning.classList.remove('hidden');
            }
        }
        
    } catch (error) {
        showError(`Error: ${error.message}`);
        console.error('Parse error:', error);
    } finally {
        setLoading(false);
    }
}

// Display parse preview
function displayParsePreview(data) {
    const previewDiv = document.getElementById('parse-preview');
    const infoDiv = document.getElementById('parse-info');
    const sampleDiv = document.getElementById('parse-sample');
    
    infoDiv.innerHTML = `
        <div class="result-card">
            <h3>Parse Information</h3>
            <div class="stat">
                <span class="stat-label">Parser:</span>
                <span class="stat-value">${data.metadata.module_name}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Total Rows:</span>
                <span class="stat-value">${data.texts.length}</span>
            </div>
            ${data.metadata.pattern ? `
                <div class="stat">
                    <span class="stat-label">Pattern:</span>
                    <span class="stat-value"><code>${data.metadata.pattern_display || data.metadata.pattern}</code></span>
                </div>
            ` : ''}
        </div>
    `;
    
    // Display sample rows
    let sampleHTML = '<table style="width: 100%; border-collapse: collapse;"><thead><tr><th style="padding: 0.5rem; border: 1px solid var(--border-color);">Index</th><th style="padding: 0.5rem; border: 1px solid var(--border-color);">Text Preview</th></tr></thead><tbody>';
    
    data.preview_rows.forEach(row => {
        sampleHTML += `
            <tr>
                <td style="padding: 0.5rem; border: 1px solid var(--border-color);">${row.index}</td>
                <td style="padding: 0.5rem; border: 1px solid var(--border-color);">${row.text}</td>
            </tr>
        `;
    });
    
    sampleHTML += '</tbody></table>';
    sampleDiv.innerHTML = sampleHTML;
    
    previewDiv.classList.remove('hidden');
}

// Handle analyze explorer
async function handleAnalyzeExplorer() {
    if (!currentParseData) {
        showError('Please parse a file first');
        return;
    }
    
    hideError();
    setLoading(true);
    
    const formData = new FormData();
    formData.append('texts', JSON.stringify(currentParseData.texts));
    formData.append('metadata_json', JSON.stringify(currentParseData.metadata));
    formData.append('ollama_endpoint', document.getElementById('explorer-ollama-endpoint').value || '');
    formData.append('embedding_model', document.getElementById('explorer-embedding-model').value || '');
    
    const nClusters = document.getElementById('explorer-n-clusters').value;
    if (nClusters) {
        formData.append('n_clusters', nClusters);
    }
    formData.append('clustering_method', document.getElementById('explorer-clustering-method').value);
    
    try {
        const response = await fetch('/api/explorer/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Analysis failed');
        }
        
        currentUnifiedData = data.unified_data;
        displayExplorerTable(data);
        updateClusterFilter(data.cluster_ids);
        
        // Display distribution visualizations
        displayDistributionCharts(data);
        
        // Add visualizations if available
        if (typeof addVisualizations === 'function' && data.coords_3d && data.coords_3d.length > 0) {
            // Create visualization data structure
            const vizData = {
                coords_3d: data.coords_3d,
                labels: data.cluster_ids,
                sentences: data.sentences || currentParseData.texts,
                overall_sentiment: null,
                cluster_sentiments: data.metadata.cluster_stats || {}
            };
            
            // Add visualization section to explorer results
            const explorerResults = document.getElementById('explorer-results');
            if (explorerResults) {
                let vizHTML = explorerResults.querySelector('#explorer-visualizations');
                if (!vizHTML) {
                    vizHTML = document.createElement('div');
                    vizHTML.id = 'explorer-visualizations';
                    vizHTML.className = 'card';
                    explorerResults.appendChild(vizHTML);
                }
                vizHTML.innerHTML = '<h2><i class="fas fa-chart-line"></i> 3D Visualizations</h2><div id="explorer-viz-section"></div>';
                
                // Temporarily set visualizations-section for addVisualizations
                const originalVizSection = document.getElementById('visualizations-section');
                const tempVizSection = document.createElement('div');
                tempVizSection.id = 'visualizations-section';
                document.getElementById('explorer-viz-section').appendChild(tempVizSection);
                
                addVisualizations(vizData);
                
                // Move visualization to explorer section
                const vizContent = tempVizSection.innerHTML;
                document.getElementById('explorer-viz-section').innerHTML = vizContent;
            }
        }
        
    } catch (error) {
        showError(`Error: ${error.message}`);
        console.error('Analysis error:', error);
    } finally {
        setLoading(false);
    }
}

// Display explorer table
function displayExplorerTable(data) {
    const container = document.getElementById('explorer-table-container');
    const resultsDiv = document.getElementById('explorer-results');
    
    if (!data.unified_data || data.unified_data.length === 0) {
        container.innerHTML = '<p>No data to display</p>';
        resultsDiv.classList.remove('hidden');
        return;
    }
    
    // Add summary statistics above table
    const summaryHTML = `
        <div class="card" style="margin-bottom: 1rem; padding: 1rem;">
            <h3 style="margin-bottom: 0.5rem; font-size: 1rem;"><i class="fas fa-info-circle"></i> Dataset Summary</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; font-size: 0.875rem;">
                <div>
                    <strong>Total Rows:</strong> ${data.num_rows}
                </div>
                <div>
                    <strong>Clusters:</strong> ${data.num_clusters}
                </div>
                <div>
                    <strong>Columns:</strong> ${Object.keys(data.unified_data[0] || {}).length}
                </div>
            </div>
        </div>
    `;
    
    // Build table
    let tableHTML = '<table id="explorer-table" style="width: 100%; border-collapse: collapse; font-size: 0.875rem;"><thead><tr>';
    
    // Column headers
    const columns = [
        'sentence_index', 'raw_text', 'cluster_id',
        'word_freq_1', 'word_freq_1_value',
        'word_freq_2', 'word_freq_2_value',
        'word_freq_3', 'word_freq_3_value',
        'tfidf_feature_1', 'tfidf_feature_1_value',
        'tfidf_feature_2', 'tfidf_feature_2_value',
        'tfidf_feature_3', 'tfidf_feature_3_value'
    ];
    
    const columnLabels = {
        'sentence_index': 'Index',
        'raw_text': 'Text',
        'cluster_id': 'Cluster',
        'word_freq_1': 'Word 1',
        'word_freq_1_value': 'Freq 1',
        'word_freq_2': 'Word 2',
        'word_freq_2_value': 'Freq 2',
        'word_freq_3': 'Word 3',
        'word_freq_3_value': 'Freq 3',
        'tfidf_feature_1': 'TF-IDF 1',
        'tfidf_feature_1_value': 'Score 1',
        'tfidf_feature_2': 'TF-IDF 2',
        'tfidf_feature_2_value': 'Score 2',
        'tfidf_feature_3': 'TF-IDF 3',
        'tfidf_feature_3_value': 'Score 3'
    };
    
    columns.forEach(col => {
        tableHTML += `<th style="padding: 0.5rem; border: 1px solid var(--border-color); background: var(--bg-color); position: sticky; top: 0;">${columnLabels[col] || col}</th>`;
    });
    
    tableHTML += '</tr></thead><tbody>';
    
    // Filter data
    let filteredData = filterTableData(data.unified_data);
    
    // Paginate
    const startIdx = (currentTablePage - 1) * rowsPerPage;
    const endIdx = startIdx + rowsPerPage;
    const pageData = filteredData.slice(startIdx, endIdx);
    
    pageData.forEach((row, idx) => {
        const globalIdx = startIdx + idx;
        tableHTML += `<tr data-index="${globalIdx}" data-cluster="${row.cluster_id}" style="cursor: pointer;" onclick="highlightTableRow(${globalIdx})">`;
        
        columns.forEach(col => {
            let value = row[col];
            if (col === 'raw_text' && value && value.length > 100) {
                value = value.substring(0, 100) + '...';
            }
            if (value === null || value === undefined) {
                value = '';
            }
            if (typeof value === 'object') {
                value = JSON.stringify(value);
            }
            tableHTML += `<td style="padding: 0.5rem; border: 1px solid var(--border-color);">${escapeHtml(String(value))}</td>`;
        });
        
        tableHTML += '</tr>';
    });
    
    tableHTML += '</tbody></table>';
    container.innerHTML = summaryHTML + tableHTML;
    
    // Update pagination
    updatePagination(filteredData.length);
    
    resultsDiv.classList.remove('hidden');
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

// Filter table data
function filterTableData(data) {
    let filtered = [...data];
    
    // Cluster filter
    const clusterFilter = document.getElementById('cluster-filter');
    const selectedClusters = Array.from(clusterFilter.selectedOptions).map(opt => opt.value);
    if (selectedClusters.length > 0 && !selectedClusters.includes('all')) {
        filtered = filtered.filter(row => selectedClusters.includes(String(row.cluster_id)));
    }
    
    // Search filter
    const searchTerm = document.getElementById('table-search').value.toLowerCase();
    if (searchTerm) {
        filtered = filtered.filter(row => {
            const text = String(row.raw_text || '').toLowerCase();
            return text.includes(searchTerm);
        });
    }
    
    return filtered;
}

// Filter table
function filterTable() {
    if (currentUnifiedData) {
        currentTablePage = 1;
        displayExplorerTable({ unified_data: currentUnifiedData });
    }
}

// Update cluster filter dropdown
function updateClusterFilter(clusterIds) {
    const clusterFilter = document.getElementById('cluster-filter');
    const uniqueClusters = [...new Set(clusterIds)].sort((a, b) => a - b);
    
    clusterFilter.innerHTML = '<option value="all">All Clusters</option>';
    uniqueClusters.forEach(clusterId => {
        const option = document.createElement('option');
        option.value = clusterId;
        option.textContent = `Cluster ${clusterId}`;
        clusterFilter.appendChild(option);
    });
}

// Update pagination
function updatePagination(totalRows) {
    const paginationDiv = document.getElementById('explorer-pagination');
    const totalPages = Math.ceil(totalRows / rowsPerPage);
    
    if (totalPages <= 1) {
        paginationDiv.innerHTML = '';
        return;
    }
    
    let paginationHTML = `<span>Page ${currentTablePage} of ${totalPages} (${totalRows} rows)</span> `;
    
    if (currentTablePage > 1) {
        paginationHTML += `<button class="btn" onclick="changeTablePage(${currentTablePage - 1})">Previous</button> `;
    }
    
    if (currentTablePage < totalPages) {
        paginationHTML += `<button class="btn" onclick="changeTablePage(${currentTablePage + 1})">Next</button>`;
    }
    
    paginationDiv.innerHTML = paginationHTML;
}

// Change table page
function changeTablePage(page) {
    currentTablePage = page;
    if (currentUnifiedData) {
        displayExplorerTable({ unified_data: currentUnifiedData });
    }
}

// Highlight table row
function highlightTableRow(index) {
    // Remove previous highlights
    document.querySelectorAll('#explorer-table tbody tr').forEach(row => {
        row.style.backgroundColor = '';
    });
    
    // Highlight selected row
    const row = document.querySelector(`#explorer-table tbody tr[data-index="${index}"]`);
    if (row) {
        row.style.backgroundColor = 'rgba(99, 102, 241, 0.1)';
        row.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// Handle export CSV
async function handleExportCSV() {
    if (!currentUnifiedData) {
        showError('No data to export');
        return;
    }
    
    const clusterFilter = document.getElementById('cluster-filter');
    const selectedClusters = Array.from(clusterFilter.selectedOptions).map(opt => opt.value);
    let clusterIds = null;
    if (selectedClusters.length > 0 && !selectedClusters.includes('all')) {
        clusterIds = selectedClusters.map(id => parseInt(id));
    }
    
    const formData = new FormData();
    formData.append('unified_data_json', JSON.stringify(currentUnifiedData));
    if (clusterIds) {
        formData.append('cluster_ids', JSON.stringify(clusterIds));
    }
    
    try {
        const response = await fetch('/api/explorer/export', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Export failed');
        }
        
        // Download file
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'explorer_export.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
    } catch (error) {
        showError(`Export error: ${error.message}`);
        console.error('Export error:', error);
    }
}

// Display distribution charts (Kaggle-style)
function displayDistributionCharts(data) {
    const explorerResults = document.getElementById('explorer-results');
    if (!explorerResults) return;
    
    const columnStats = data.metadata?.column_statistics || {};
    if (Object.keys(columnStats).length === 0) return;
    
    // Create distribution section
    let distHTML = explorerResults.querySelector('#explorer-distributions');
    if (!distHTML) {
        distHTML = document.createElement('div');
        distHTML.id = 'explorer-distributions';
        distHTML.className = 'card';
        // Insert before visualizations if exists, otherwise append
        const vizSection = explorerResults.querySelector('#explorer-visualizations');
        if (vizSection) {
            explorerResults.insertBefore(distHTML, vizSection);
        } else {
            explorerResults.appendChild(distHTML);
        }
    }
    
    distHTML.innerHTML = '<h2><i class="fas fa-chart-bar"></i> Column Distributions</h2><div id="distribution-charts" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; margin-top: 1rem;"></div>';
    
    const chartsContainer = document.getElementById('distribution-charts');
    
    // Render charts for each column
    Object.entries(columnStats).forEach(([colName, stats]) => {
        if (stats.type === 'numeric') {
            renderNumericDistribution(chartsContainer, colName, stats);
        } else if (stats.type === 'categorical') {
            renderCategoricalDistribution(chartsContainer, colName, stats);
        }
    });
}

// Render numeric distribution (histogram + box plot)
function renderNumericDistribution(container, colName, stats) {
    const chartId = `chart-${colName.replace(/[^a-zA-Z0-9]/g, '-')}`;
    const cardHTML = `
        <div class="distribution-card" style="background: var(--card-bg); padding: 1.5rem; border-radius: 0.75rem; box-shadow: var(--shadow);">
            <h3 style="margin-bottom: 1rem; color: var(--primary-color);">${formatColumnName(colName)}</h3>
            <div style="margin-bottom: 0.5rem; font-size: 0.875rem; color: var(--text-secondary);">
                <div><strong>Count:</strong> ${stats.count}</div>
                <div><strong>Mean:</strong> ${stats.mean.toFixed(2)}</div>
                <div><strong>Std:</strong> ${stats.std.toFixed(2)}</div>
                <div><strong>Min:</strong> ${stats.min.toFixed(2)} | <strong>Max:</strong> ${stats.max.toFixed(2)}</div>
                <div><strong>Median:</strong> ${stats.median.toFixed(2)}</div>
                <div><strong>Q25:</strong> ${stats.q25.toFixed(2)} | <strong>Q75:</strong> ${stats.q75.toFixed(2)}</div>
            </div>
            <div id="${chartId}" style="width: 100%; height: 300px;"></div>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', cardHTML);
    
    // Create histogram
    if (typeof Plotly !== 'undefined' && stats.values && stats.values.length > 0) {
        setTimeout(() => {
            const histogram = {
                x: stats.values,
                type: 'histogram',
                marker: {
                    color: 'rgba(99, 102, 241, 0.7)',
                    line: {
                        color: 'rgba(99, 102, 241, 1)',
                        width: 1
                    }
                },
                nbinsx: Math.min(30, Math.ceil(Math.sqrt(stats.values.length)))
            };
            
            const layout = {
                title: 'Distribution',
                xaxis: { title: formatColumnName(colName) },
                yaxis: { title: 'Frequency' },
                height: 300,
                margin: { t: 40, b: 40, l: 50, r: 20 },
                showlegend: false
            };
            
            Plotly.newPlot(chartId, [histogram], layout, {responsive: true});
        }, 100);
    }
}

// Render categorical distribution (bar chart)
function renderCategoricalDistribution(container, colName, stats) {
    const chartId = `chart-${colName.replace(/[^a-zA-Z0-9]/g, '-')}`;
    const topValues = stats.top_values || [];
    
    if (topValues.length === 0) return;
    
    const cardHTML = `
        <div class="distribution-card" style="background: var(--card-bg); padding: 1.5rem; border-radius: 0.75rem; box-shadow: var(--shadow);">
            <h3 style="margin-bottom: 1rem; color: var(--primary-color);">${formatColumnName(colName)}</h3>
            <div style="margin-bottom: 0.5rem; font-size: 0.875rem; color: var(--text-secondary);">
                <div><strong>Unique Values:</strong> ${stats.unique_count}</div>
                <div><strong>Top Values:</strong> ${topValues.length} shown</div>
            </div>
            <div id="${chartId}" style="width: 100%; height: 300px;"></div>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', cardHTML);
    
    // Create bar chart
    if (typeof Plotly !== 'undefined' && topValues.length > 0) {
        setTimeout(() => {
            const labels = topValues.map(v => String(v[0]).substring(0, 30));
            const values = topValues.map(v => v[1]);
            
            const barChart = {
                x: labels,
                y: values,
                type: 'bar',
                marker: {
                    color: 'rgba(99, 102, 241, 0.7)',
                    line: {
                        color: 'rgba(99, 102, 241, 1)',
                        width: 1
                    }
                }
            };
            
            const layout = {
                title: 'Value Distribution',
                xaxis: { 
                    title: formatColumnName(colName),
                    tickangle: -45
                },
                yaxis: { title: 'Count' },
                height: 300,
                margin: { t: 40, b: 80, l: 50, r: 20 },
                showlegend: false
            };
            
            Plotly.newPlot(chartId, [barChart], layout, {responsive: true});
        }, 100);
    }
}

// Format column name for display
function formatColumnName(colName) {
    return colName
        .replace(/_/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, str => str.toUpperCase())
        .trim();
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Make functions globally available
window.changeTablePage = changeTablePage;
window.highlightTableRow = highlightTableRow;

