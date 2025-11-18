"""CLI entry point for semantic detector."""

import os
import sys
import argparse
from pathlib import Path

from semantic_detector.core.detector import SemanticDetector


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Semantic Detector - Language Fingerprinting')
    parser.add_argument('file', nargs='?', help='Text file to analyze')
    parser.add_argument('--endpoint', default='http://localhost:11434',
                       help='Ollama endpoint URL (default: http://localhost:11434)')
    parser.add_argument('--model', default='nomic-embed-text',
                       help='Embedding model name (default: nomic-embed-text)')
    parser.add_argument('--clusters', type=int, default=None,
                       help='Number of clusters (default: auto-detect)')
    parser.add_argument('--method', default='kmeans',
                       choices=['kmeans', 'dbscan', 'hierarchical'],
                       help='Clustering method (default: kmeans)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--find-style', type=int, default=None,
                       help='Find closest styles for sentence at index (0-based)')
    parser.add_argument('--style-k', type=int, default=5,
                       help='Number of closest styles to find (default: 5)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = SemanticDetector(args.endpoint, args.model)
    
    # Check available models (optional, don't fail if Ollama is not running)
    try:
        models = detector.embedding_generator.list_models()
        if models:
            print(f"[INFO] Available models: {', '.join(models)}")
            if args.model not in models:
                print(f"[WARNING] Model '{args.model}' not found. Available: {', '.join(models)}")
    except Exception:
        pass  # Ollama might not be running, that's OK
    print()
    
    if args.interactive or not args.file:
        # Interactive mode
        print("=== Semantic Detector - Interactive Mode ===")
        print("Enter text file path or 'quit' to exit")
        
        while True:
            file_path = input("\nFile path: ").strip()
            if file_path.lower() in ['quit', 'exit', 'q']:
                break
            
            if not os.path.exists(file_path):
                print(f"[ERROR] File not found: {file_path}")
                continue
            
            try:
                result = detector.process_text_file(file_path, args.clusters, args.method)
                
                # Handle style finding if requested
                if args.find_style is not None:
                    style_finder = result['style_finder']
                    sentences = result['sentences']
                    labels = result['labels']
                    
                    if args.find_style >= len(sentences):
                        print(f"[ERROR] Sentence index {args.find_style} out of range (0-{len(sentences)-1})")
                        continue
                    
                    print(f"\n{'='*60}")
                    print(f"Finding closest styles for sentence {args.find_style}")
                    print(f"{'='*60}")
                    print(f"\nQuery sentence: {sentences[args.find_style]}")
                    print(f"Cluster: {labels[args.find_style]}")
                    
                    # Find closest in cluster
                    in_cluster = style_finder.find_closest_in_cluster(args.find_style, k=args.style_k)
                    print(f"\n--- Closest styles IN cluster (excluding self) ---")
                    for i, (idx, sim) in enumerate(in_cluster, 1):
                        print(f"{i}. Similarity: {sim:.4f} | Cluster: {labels[idx]}")
                        print(f"   Text: {sentences[idx][:100]}...")
                    
                    # Find closest out of cluster
                    out_cluster = style_finder.find_closest_out_cluster(args.find_style, k=args.style_k)
                    print(f"\n--- Closest styles OUTSIDE cluster ---")
                    for i, (idx, sim) in enumerate(out_cluster, 1):
                        print(f"{i}. Similarity: {sim:.4f} | Cluster: {labels[idx]}")
                        print(f"   Text: {sentences[idx][:100]}...")
                    
            except Exception as e:
                print(f"[ERROR] Error processing file: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Process single file
        if not os.path.exists(args.file):
            print(f"[ERROR] File not found: {args.file}")
            sys.exit(1)
        
        try:
            result = detector.process_text_file(args.file, args.clusters, args.method)
            
            # If style finding is requested
            if args.find_style is not None:
                style_finder = result['style_finder']
                sentences = result['sentences']
                labels = result['labels']
                
                if args.find_style >= len(sentences):
                    print(f"[ERROR] Sentence index {args.find_style} out of range (0-{len(sentences)-1})")
                    sys.exit(1)
                
                print(f"\n{'='*60}")
                print(f"Finding closest styles for sentence {args.find_style}")
                print(f"{'='*60}")
                print(f"\nQuery sentence: {sentences[args.find_style]}")
                print(f"Cluster: {labels[args.find_style]}")
                
                # Find closest in cluster (excluding self)
                print(f"\n--- Closest styles IN cluster (excluding self) ---")
                in_cluster = style_finder.find_closest_in_cluster(args.find_style, k=args.style_k)
                for i, (idx, sim) in enumerate(in_cluster, 1):
                    print(f"{i}. Similarity: {sim:.4f} | Cluster: {labels[idx]}")
                    print(f"   Text: {sentences[idx][:100]}...")
                
                # Find closest out of cluster
                print(f"\n--- Closest styles OUTSIDE cluster ---")
                out_cluster = style_finder.find_closest_out_cluster(args.find_style, k=args.style_k)
                for i, (idx, sim) in enumerate(out_cluster, 1):
                    print(f"{i}. Similarity: {sim:.4f} | Cluster: {labels[idx]}")
                    print(f"   Text: {sentences[idx][:100]}...")
                
                # Show fingerprint summary
                print(f"\n--- Fingerprint Summary ---")
                fp_summary = detector.fingerprint_extractor.get_fingerprint_summary(sentences[args.find_style])
                print(f"Top words: {', '.join(fp_summary['top_words'][:10])}")
                print(f"Top bigrams: {', '.join(fp_summary['top_bigrams'][:5])}")
                print(f"Avg sentence length: {fp_summary['avg_sentence_length']:.1f} words")
                print(f"Vocabulary richness: {fp_summary['vocab_richness']:.3f}")
                
        except Exception as e:
            print(f"[ERROR] Error processing file: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()

