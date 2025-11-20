"""FastAPI routes."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from typing import Optional, List, Dict, Any
import os
import shutil
import csv
import io
import numpy as np

from semantic_detector.web.app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    StyleSearchRequest,
    StyleSearchResponse,
    ClusteringMethod,
    ParseRequest,
    ParseResponse,
    ExplorerAnalysisResponse,
    ExportRequest
)
from semantic_detector.web.app.core.detector import detector_service
from semantic_detector.web.app.core.config import settings
from semantic_detector.web.app.core.parsers import (
    CSVParser,
    JSONParser,
    RegexParser,
    MarkdownParser,
    PDFParser,
    UnifiedDataFormatter
)

router = APIRouter()


# Removed duplicate index route - handled in app/main.py


@router.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """Analyze text or file."""
    try:
        # Initialize detector if needed
        detector_service.initialize(
            ollama_endpoint=request.ollama_endpoint,
            embedding_model=request.embedding_model
        )
        
        # Process text or file
        if request.text:
            result = detector_service.process_text(
                text=request.text,
                n_clusters=request.n_clusters,
                clustering_method=request.clustering_method.value
            )
        elif request.file_path:
            if not os.path.exists(request.file_path):
                raise HTTPException(status_code=404, detail="File not found")
            result = detector_service.process_file(
                file_path=request.file_path,
                n_clusters=request.n_clusters,
                clustering_method=request.clustering_method.value
            )
        else:
            raise HTTPException(status_code=400, detail="Either text or file_path must be provided")
        
        # Format response
        cluster_stats = {
            str(k): {
                'cluster_id': k,
                'count': v['count'],
                'avg_length': float(v['avg_length']),
                'sample_texts': v.get('sample_texts', [])
            }
            for k, v in result.get('cluster_stats', {}).items()
        }
        
        fingerprint_summaries = {
            str(k): {
                'top_words': v['top_words'],
                'top_bigrams': v['top_bigrams'],
                'top_trigrams': v['top_trigrams'],
                'avg_sentence_length': float(v['avg_sentence_length']),
                'vocab_richness': float(v['vocab_richness']),
                'punctuation_style': {kk: float(vv) for kk, vv in v.get('punctuation_style', {}).items()}
            }
            for k, v in result.get('fingerprint_summaries', {}).items()
        }
        
        return AnalysisResponse(
            success=True,
            message="Analysis completed successfully",
            num_sentences=len(result.get('sentences', [])),
            num_clusters=len(set(result.get('labels', []))),
            cluster_stats=cluster_stats,
            fingerprint_summaries=fingerprint_summaries,
            output_files=result.get('output_files', {}),
            metadata=result.get('metadata', {}),
            tfidf_features=result.get('cluster_tfidf_features', {}),
            sentiment_analysis=result.get('sentiment_analysis', []),
            cluster_sentiments=result.get('cluster_sentiments', {}),
            overall_sentiment=result.get('overall_sentiment', {}),
            overall_person_score=result.get('overall_person_score', 50.0),
            coords_3d=result.get('coords_3d', []).tolist() if hasattr(result.get('coords_3d'), 'tolist') else result.get('coords_3d', []),
            labels=result.get('labels', []).tolist() if hasattr(result.get('labels'), 'tolist') else result.get('labels', []),
            sentences=result.get('sentences', [])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/upload", response_model=AnalysisResponse)
async def upload_file(
    file: UploadFile = File(...),
    ollama_endpoint: Optional[str] = Form(None),
    embedding_model: Optional[str] = Form(None),
    n_clusters: Optional[int] = Form(None),
    clustering_method: ClusteringMethod = Form(ClusteringMethod.KMEANS)
):
    """Upload and analyze file."""
    try:
        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1]
        if file_ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File extension {file_ext} not allowed. Allowed: {settings.allowed_extensions}"
            )
        
        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Initialize detector
            detector_service.initialize(
                ollama_endpoint=ollama_endpoint,
                embedding_model=embedding_model
            )
            
            # Process file
            result = detector_service.process_file(
                file_path=file_path,
                n_clusters=n_clusters,
                clustering_method=clustering_method.value
            )
            
            # Format response (same as analyze_text)
            cluster_stats = {
                str(k): {
                    'cluster_id': k,
                    'count': v['count'],
                    'avg_length': float(v['avg_length']),
                    'sample_texts': v.get('sample_texts', [])
                }
                for k, v in result.get('cluster_stats', {}).items()
            }
            
            fingerprint_summaries = {
                str(k): {
                    'top_words': v['top_words'],
                    'top_bigrams': v['top_bigrams'],
                    'top_trigrams': v['top_trigrams'],
                    'avg_sentence_length': float(v['avg_sentence_length']),
                    'vocab_richness': float(v['vocab_richness']),
                    'punctuation_style': {kk: float(vv) for kk, vv in v.get('punctuation_style', {}).items()}
                }
                for k, v in result.get('fingerprint_summaries', {}).items()
            }
            
            return AnalysisResponse(
                success=True,
                message="Analysis completed successfully",
                num_sentences=len(result.get('sentences', [])),
                num_clusters=len(set(result.get('labels', []))),
                cluster_stats=cluster_stats,
                fingerprint_summaries=fingerprint_summaries,
                output_files=result.get('output_files', {}),
                metadata=result.get('metadata', {}),
                tfidf_features=result.get('cluster_tfidf_features', {}),
                sentiment_analysis=result.get('sentiment_analysis', []),
                cluster_sentiments=result.get('cluster_sentiments', {}),
                overall_sentiment=result.get('overall_sentiment', {}),
                overall_person_score=result.get('overall_person_score', 50.0),
                coords_3d=result.get('coords_3d', []).tolist() if hasattr(result.get('coords_3d'), 'tolist') else result.get('coords_3d', []),
                labels=result.get('labels', []).tolist() if hasattr(result.get('labels'), 'tolist') else result.get('labels', []),
                sentences=result.get('sentences', [])
            )
        
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/style-search", response_model=StyleSearchResponse)
async def search_style(request: StyleSearchRequest):
    """Search for similar styles."""
    try:
        results = detector_service.get_style_matches(
            sentence_index=request.sentence_index,
            k=request.k,
            in_cluster_only=request.in_cluster_only,
            out_cluster_only=request.out_cluster_only
        )
        
        return StyleSearchResponse(
            query_index=results['query_index'],
            query_text=results['query_text'],
            query_cluster=results['query_cluster'],
            in_cluster_matches=results.get('in_cluster', []),
            out_cluster_matches=results.get('out_cluster', []),
            fingerprint_summary=results['fingerprint_summary']
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/models")
async def list_models():
    """List available Ollama models."""
    try:
        detector_service.initialize()
        models = detector_service.detector.embedding_generator.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Explorer endpoints
@router.post("/api/explorer/parse", response_model=ParseResponse)
async def parse_file(
    file: UploadFile = File(...),
    format_type: str = Form(...),
    text_column: Optional[str] = Form(None),
    jsonpath: Optional[str] = Form(None),
    regex_pattern: Optional[str] = Form(None),
    use_capture_group: bool = Form(False),
    markdown_mode: Optional[str] = Form("sections")
):
    """Parse uploaded file (CSV/JSON/regex/markdown/PDF) and return preview."""
    try:
        # Read file content
        content = await file.read()
        parsed_data = None
        
        if format_type == 'csv':
            file_content = content.decode('utf-8')
            if not text_column:
                raise HTTPException(status_code=400, detail="text_column required for CSV")
            parser = CSVParser()
            parsed_data = parser.parse(file_content, text_column)
        
        elif format_type == 'json':
            file_content = content.decode('utf-8')
            if not jsonpath:
                raise HTTPException(status_code=400, detail="jsonpath required for JSON")
            parser = JSONParser()
            parsed_data = parser.parse(file_content, jsonpath)
        
        elif format_type == 'regex':
            file_content = content.decode('utf-8')
            if not regex_pattern:
                raise HTTPException(status_code=400, detail="regex_pattern required for regex")
            parser = RegexParser()
            parsed_data = parser.parse(file_content, regex_pattern, use_capture_group)
        
        elif format_type == 'markdown':
            file_content = content.decode('utf-8')
            parser = MarkdownParser()
            parsed_data = parser.parse(file_content, extract_mode=markdown_mode or "sections")
        
        elif format_type == 'pdf':
            # PDF is binary, don't decode
            parser = PDFParser()
            parsed_data = parser.parse(content, extract_mode=markdown_mode or "sections")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown format_type: {format_type}")
        
        # Create preview (first 10 rows)
        preview_rows = []
        texts = parsed_data['texts']
        metadata_list = parsed_data['metadata'].get('row_metadata', [])
        
        for i in range(min(10, len(texts))):
            preview_row = {
                'index': i,
                'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i]
            }
            if i < len(metadata_list):
                preview_row['metadata'] = metadata_list[i]
            preview_rows.append(preview_row)
        
        # Build message with warning if PDF
        message = f"Parsed {len(texts)} rows using {parsed_data['metadata']['module_name']}"
        if parsed_data['metadata'].get('pdf_warning'):
            message += " (Note: PDFs are converted to markdown format for processing)"
        
        return ParseResponse(
            success=True,
            message=message,
            texts=texts,
            metadata=parsed_data['metadata'],
            preview_rows=preview_rows
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/explorer/columns")
async def get_csv_columns(file: UploadFile = File(...)):
    """Get column names from CSV file for dropdown."""
    try:
        content = await file.read()
        file_content = content.decode('utf-8')
        
        reader = csv.DictReader(io.StringIO(file_content))
        columns = list(next(reader, {}).keys())
        
        return {"columns": columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/explorer/analyze", response_model=ExplorerAnalysisResponse)
async def analyze_structured_data(
    texts: str = Form(...),  # JSON array of texts
    metadata_json: str = Form(...),  # JSON metadata
    ollama_endpoint: Optional[str] = Form(None),
    embedding_model: Optional[str] = Form(None),
    n_clusters: Optional[int] = Form(None),
    clustering_method: ClusteringMethod = Form(ClusteringMethod.KMEANS)
):
    """Run clustering analysis on parsed structured data."""
    try:
        import json
        texts_list = json.loads(texts)
        metadata = json.loads(metadata_json)
        
        if not isinstance(texts_list, list):
            raise HTTPException(status_code=400, detail="texts must be a JSON array")
        
        # Process structured data
        result = detector_service.process_structured_data(
            texts=texts_list,
            n_clusters=n_clusters,
            clustering_method=clustering_method.value,
            ollama_endpoint=ollama_endpoint,
            embedding_model=embedding_model
        )
        
        # Format unified data
        formatter = UnifiedDataFormatter()
        unified_data = formatter.format_for_analysis(
            texts=result['sentences'],
            cluster_ids=result['labels'].tolist() if hasattr(result['labels'], 'tolist') else list(result['labels']),
            word_frequencies=result.get('word_frequencies', []),
            tfidf_features=result.get('tfidf_features_per_text', []),
            metadata=metadata.get('row_metadata', [])
        )
        
        # Calculate column statistics for distributions
        column_stats = calculate_column_statistics(unified_data)
        
        return ExplorerAnalysisResponse(
            success=True,
            message="Analysis completed successfully",
            unified_data=unified_data,
            num_rows=len(unified_data),
            num_clusters=len(set(result['labels'])),
            cluster_ids=result['labels'].tolist() if hasattr(result['labels'], 'tolist') else list(result['labels']),
            coords_3d=result.get('coords_3d', []).tolist() if hasattr(result.get('coords_3d'), 'tolist') else result.get('coords_3d', []),
            sentences=result.get('sentences', []),
            metadata={
                'parser_metadata': metadata,
                'cluster_stats': {
                    str(k): {
                        'cluster_id': k,
                        'count': v['count'],
                        'avg_length': float(v['avg_length'])
                    }
                    for k, v in result.get('cluster_stats', {}).items()
                },
                'column_statistics': column_stats
            }
        )


def calculate_column_statistics(unified_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for each column for distribution visualization."""
    import numpy as np
    
    stats = {}
    
    if not unified_data:
        return stats
    
    # Get all numeric columns
    numeric_columns = []
    text_columns = []
    categorical_columns = []
    
    # Sample first row to identify column types
    sample_row = unified_data[0]
    
    for col_name in sample_row.keys():
        if col_name in ['sentence_index', 'cluster_id'] or '_value' in col_name:
            numeric_columns.append(col_name)
        elif col_name == 'raw_text':
            text_columns.append(col_name)
        elif col_name in ['word_freq_1', 'word_freq_2', 'tfidf_feature_1', 'tfidf_feature_2']:
            categorical_columns.append(col_name)
    
    # Calculate statistics for numeric columns
    for col in numeric_columns:
        values = []
        for row in unified_data:
            val = row.get(col)
            if val is not None:
                try:
                    num_val = float(val)
                    if not np.isnan(num_val):
                        values.append(num_val)
                except (ValueError, TypeError):
                    pass
        
        if values:
            values_array = np.array(values)
            stats[col] = {
                'type': 'numeric',
                'count': len(values),
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
                'q25': float(np.percentile(values_array, 25)),
                'q75': float(np.percentile(values_array, 75)),
                'values': values[:1000] if len(values) > 1000 else values  # Sample for large datasets
            }
    
    # Calculate statistics for categorical columns
    for col in categorical_columns:
        value_counts = {}
        for row in unified_data:
            val = row.get(col)
            if val:
                value_counts[val] = value_counts.get(val, 0) + 1
        
        if value_counts:
            sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
            stats[col] = {
                'type': 'categorical',
                'unique_count': len(value_counts),
                'top_values': sorted_counts[:20],  # Top 20 values
                'value_counts': dict(sorted_counts[:50])  # Top 50 for distribution
            }
    
    # Text length statistics
    text_lengths = []
    for row in unified_data:
        text = row.get('raw_text', '')
        if text:
            text_lengths.append(len(text))
    
    if text_lengths:
        text_lengths_array = np.array(text_lengths)
        stats['text_length'] = {
            'type': 'numeric',
            'count': len(text_lengths),
            'mean': float(np.mean(text_lengths_array)),
            'std': float(np.std(text_lengths_array)),
            'min': float(np.min(text_lengths_array)),
            'max': float(np.max(text_lengths_array)),
            'median': float(np.median(text_lengths_array)),
            'q25': float(np.percentile(text_lengths_array, 25)),
            'q75': float(np.percentile(text_lengths_array, 75)),
            'values': text_lengths[:1000] if len(text_lengths) > 1000 else text_lengths
        }
    
    # Cluster distribution
    cluster_counts = {}
    for row in unified_data:
        cluster_id = row.get('cluster_id')
        if cluster_id is not None:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    if cluster_counts:
        stats['cluster_id'] = {
            'type': 'categorical',
            'unique_count': len(cluster_counts),
            'value_counts': cluster_counts,
            'top_values': sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        }
    
    return stats
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/explorer/export")
async def export_to_csv(
    unified_data_json: str = Form(...),
    cluster_ids: Optional[str] = Form(None)
):
    """Export unified data as CSV."""
    try:
        import json
        unified_data = json.loads(unified_data_json)
        
        # Filter by cluster IDs if provided
        if cluster_ids:
            cluster_filter = json.loads(cluster_ids)
            if isinstance(cluster_filter, list):
                unified_data = [
                    row for row in unified_data
                    if row.get('cluster_id') in cluster_filter
                ]
        
        if not unified_data:
            raise HTTPException(status_code=400, detail="No data to export")
        
        # Get all possible column names
        all_columns = set()
        for row in unified_data:
            all_columns.update(row.keys())
        
        # Order columns logically
        column_order = [
            'sentence_index', 'raw_text', 'cluster_id',
            'word_freq_1', 'word_freq_1_value',
            'word_freq_2', 'word_freq_2_value',
            'word_freq_3', 'word_freq_3_value',
            'word_freq_4', 'word_freq_4_value',
            'word_freq_5', 'word_freq_5_value',
            'word_freq_6', 'word_freq_6_value',
            'word_freq_7', 'word_freq_7_value',
            'word_freq_8', 'word_freq_8_value',
            'word_freq_9', 'word_freq_9_value',
            'word_freq_10', 'word_freq_10_value',
            'tfidf_feature_1', 'tfidf_feature_1_value',
            'tfidf_feature_2', 'tfidf_feature_2_value',
            'tfidf_feature_3', 'tfidf_feature_3_value',
            'tfidf_feature_4', 'tfidf_feature_4_value',
            'tfidf_feature_5', 'tfidf_feature_5_value',
            'tfidf_feature_6', 'tfidf_feature_6_value',
            'tfidf_feature_7', 'tfidf_feature_7_value',
            'tfidf_feature_8', 'tfidf_feature_8_value',
            'tfidf_feature_9', 'tfidf_feature_9_value',
            'tfidf_feature_10', 'tfidf_feature_10_value',
            'source_metadata'
        ]
        
        # Add any remaining columns
        remaining = sorted(all_columns - set(column_order))
        column_order.extend(remaining)
        
        # Create CSV
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=column_order, extrasaction='ignore')
        writer.writeheader()
        
        for row in unified_data:
            # Convert source_metadata to string if dict
            if 'source_metadata' in row and isinstance(row['source_metadata'], dict):
                row['source_metadata'] = json.dumps(row['source_metadata'])
            writer.writerow(row)
        
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=explorer_export.csv"}
        )
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

