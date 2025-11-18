"""FastAPI routes."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional
import os
import shutil

from semantic_detector.web.app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    StyleSearchRequest,
    StyleSearchResponse,
    ClusteringMethod
)
from semantic_detector.web.app.core.detector import detector_service
from semantic_detector.web.app.core.config import settings

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

