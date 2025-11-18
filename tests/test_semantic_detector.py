"""
Test suite for semantic detector components.
Tests all major functionality without requiring Ollama.
"""

import unittest
import numpy as np
from semantic_detector.sentence_splitter import SentenceSplitter
from semantic_detector.language_fingerprint import LanguageFingerprint
from semantic_detector.clustering import SpeakerClusterer
from semantic_detector.style_finder import StyleFinder


class TestSentenceSplitter(unittest.TestCase):
    """Test sentence splitting functionality."""
    
    def setUp(self):
        self.splitter = SentenceSplitter()
    
    def test_basic_split(self):
        """Test basic sentence splitting."""
        text = "Hello world. This is a test. Another sentence here."
        sentences = self.splitter.split_by_period(text)
        self.assertGreater(len(sentences), 0)
        self.assertIn("Hello world", sentences[0])
    
    def test_file_split(self):
        """Test file splitting."""
        # Create a temporary test file
        test_text = "First sentence. Second sentence. Third sentence."
        with open('test_temp.txt', 'w') as f:
            f.write(test_text)
        
        try:
            sentences = self.splitter.split_file('test_temp.txt')
            self.assertGreaterEqual(len(sentences), 2)
        finally:
            import os
            if os.path.exists('test_temp.txt'):
                os.remove('test_temp.txt')
    
    def test_long_sentence_handling(self):
        """Test that long sentences are handled properly."""
        long_text = "A" * 1000 + ". Short."
        sentences = self.splitter.split_by_period(long_text)
        self.assertGreater(len(sentences), 0)


class TestLanguageFingerprint(unittest.TestCase):
    """Test language fingerprinting functionality."""
    
    def setUp(self):
        self.fp = LanguageFingerprint()
        self.test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence."
    
    def test_extract_features(self):
        """Test feature extraction."""
        features = self.fp.extract_features(self.test_text)
        self.assertIsInstance(features, dict)
        self.assertIn('word_count', features)
        self.assertIn('char_count', features)
        self.assertIn('avg_sentence_length', features)
    
    def test_extract_fingerprint_vector(self):
        """Test fingerprint vector extraction."""
        vector = self.fp.extract_fingerprint_vector(self.test_text)
        self.assertIsInstance(vector, np.ndarray)
        self.assertGreater(len(vector), 0)
    
    def test_extract_word_distribution(self):
        """Test word distribution extraction."""
        dist = self.fp.extract_word_distribution(self.test_text, top_n=10)
        self.assertIsInstance(dist, dict)
        self.assertLessEqual(len(dist), 10)
        if dist:
            # Check that values are frequencies (0-1)
            for freq in dist.values():
                self.assertGreaterEqual(freq, 0)
                self.assertLessEqual(freq, 1)
    
    def test_extract_ngram_distribution(self):
        """Test n-gram distribution extraction."""
        bigrams = self.fp.extract_ngram_distribution(self.test_text, n=2, top_n=5)
        self.assertIsInstance(bigrams, dict)
        if bigrams:
            for ngram, freq in bigrams.items():
                self.assertIsInstance(ngram, tuple)
                self.assertEqual(len(ngram), 2)
                self.assertGreaterEqual(freq, 0)
                self.assertLessEqual(freq, 1)
    
    def test_extract_contextual_features(self):
        """Test contextual feature extraction."""
        contextual = self.fp.extract_contextual_features(self.test_text)
        self.assertIn('word_distribution', contextual)
        self.assertIn('bigram_distribution', contextual)
        self.assertIn('trigram_distribution', contextual)
        self.assertIn('features', contextual)
    
    def test_get_fingerprint_summary(self):
        """Test fingerprint summary generation."""
        summary = self.fp.get_fingerprint_summary(self.test_text)
        self.assertIn('top_words', summary)
        self.assertIn('top_bigrams', summary)
        self.assertIn('top_trigrams', summary)
        self.assertIn('avg_sentence_length', summary)
        self.assertIn('vocab_richness', summary)
    
    def test_batch_extraction(self):
        """Test batch fingerprint extraction."""
        texts = [
            "First sentence here.",
            "Second sentence with different words.",
            "Third sentence for testing."
        ]
        fingerprints = self.fp.extract_fingerprints_batch(texts)
        self.assertEqual(len(fingerprints), len(texts))
        self.assertEqual(fingerprints.shape[0], len(texts))


class TestClustering(unittest.TestCase):
    """Test clustering functionality."""
    
    def setUp(self):
        self.clusterer = SpeakerClusterer(n_clusters=2, method='kmeans')
        # Create dummy data
        np.random.seed(42)
        self.features = np.random.rand(20, 10)
        self.embeddings = np.random.rand(20, 50)
    
    def test_fit_with_features_only(self):
        """Test clustering with features only."""
        labels = self.clusterer.fit(self.features, None)
        self.assertEqual(len(labels), len(self.features))
        self.assertGreater(len(set(labels)), 0)
    
    def test_fit_with_embeddings_only(self):
        """Test clustering with embeddings only."""
        labels = self.clusterer.fit(None, self.embeddings)
        self.assertEqual(len(labels), len(self.embeddings))
        self.assertGreater(len(set(labels)), 0)
    
    def test_fit_with_both(self):
        """Test clustering with both features and embeddings."""
        labels = self.clusterer.fit(self.features, self.embeddings)
        self.assertEqual(len(labels), len(self.features))
        self.assertGreater(len(set(labels)), 0)
    
    def test_get_cluster_stats(self):
        """Test cluster statistics generation."""
        labels = self.clusterer.fit(self.features, self.embeddings)
        texts = [f"Sentence {i}" for i in range(len(labels))]
        stats = self.clusterer.get_cluster_stats(texts)
        self.assertIsInstance(stats, dict)
        self.assertGreater(len(stats), 0)


class TestStyleFinder(unittest.TestCase):
    """Test style finder functionality."""
    
    def setUp(self):
        np.random.seed(42)
        # Create dummy data with clear clusters
        n_samples = 20
        self.embeddings = np.random.rand(n_samples, 50)
        self.fingerprints = np.random.rand(n_samples, 20)
        
        # Create labels with 2 clusters
        self.labels = np.array([0] * 10 + [1] * 10)
        
        self.style_finder = StyleFinder(self.embeddings, self.fingerprints, self.labels)
    
    def test_find_closest_styles(self):
        """Test finding closest styles."""
        results = self.style_finder.find_closest_styles(0, k=5)
        self.assertLessEqual(len(results), 5)
        for idx, sim, info in results:
            self.assertIsInstance(idx, int)
            self.assertIsInstance(sim, float)
            self.assertIsInstance(info, str)
    
    def test_find_closest_in_cluster(self):
        """Test finding closest styles in cluster."""
        results = self.style_finder.find_closest_in_cluster(0, k=5)
        self.assertLessEqual(len(results), 5)
        for idx, sim in results:
            self.assertEqual(self.labels[idx], self.labels[0])
            self.assertNotEqual(idx, 0)  # Should exclude self
    
    def test_find_closest_out_cluster(self):
        """Test finding closest styles outside cluster."""
        results = self.style_finder.find_closest_out_cluster(0, k=5)
        self.assertLessEqual(len(results), 5)
        for idx, sim in results:
            self.assertNotEqual(self.labels[idx], self.labels[0])
    
    def test_get_cluster_style_profile(self):
        """Test cluster style profile generation."""
        profile = self.style_finder.get_cluster_style_profile(0)
        self.assertIn('cluster_id', profile)
        self.assertIn('size', profile)
        self.assertIn('avg_embedding', profile)
        self.assertIn('avg_fingerprint', profile)
        self.assertEqual(profile['cluster_id'], 0)
        self.assertEqual(profile['size'], 10)
    
    def test_compare_clusters(self):
        """Test cluster comparison."""
        comparison = self.style_finder.compare_clusters(0, 1)
        self.assertIn('embedding_similarity', comparison)
        self.assertIn('fingerprint_similarity', comparison)
        self.assertIn('combined_similarity', comparison)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test a simplified version of the full pipeline."""
        # Create test data
        texts = [
            "The quick brown fox jumps.",
            "A lazy dog sleeps peacefully.",
            "Technology advances rapidly today.",
            "Artificial intelligence transforms industries.",
            "Natural language processing is fascinating.",
            "Machine learning models understand context."
        ]
        
        # Test sentence splitting
        splitter = SentenceSplitter()
        all_sentences = []
        for text in texts:
            sentences = splitter.split_by_period(text)
            all_sentences.extend(sentences)
        
        self.assertGreater(len(all_sentences), 0)
        
        # Test fingerprint extraction
        fp = LanguageFingerprint()
        fingerprints = fp.extract_fingerprints_batch(all_sentences)
        self.assertEqual(len(fingerprints), len(all_sentences))
        
        # Test clustering
        # Create dummy embeddings since we don't have Ollama
        np.random.seed(42)
        embeddings = np.random.rand(len(all_sentences), 50)
        
        clusterer = SpeakerClusterer(n_clusters=2, method='kmeans')
        labels = clusterer.fit(fingerprints, embeddings)
        self.assertEqual(len(labels), len(all_sentences))
        
        # Test style finder
        style_finder = StyleFinder(embeddings, fingerprints, labels)
        results = style_finder.find_closest_styles(0, k=3)
        self.assertLessEqual(len(results), 3)


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Semantic Detector Test Suite")
    print("=" * 60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSentenceSplitter))
    suite.addTests(loader.loadTestsFromTestCase(TestLanguageFingerprint))
    suite.addTests(loader.loadTestsFromTestCase(TestClustering))
    suite.addTests(loader.loadTestsFromTestCase(TestStyleFinder))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)

