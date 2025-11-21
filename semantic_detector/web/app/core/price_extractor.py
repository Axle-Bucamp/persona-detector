"""Price extraction from text content."""

import re
from typing import List, Dict, Optional, Tuple


class PriceExtractor:
    """Extract monetary values and currencies from text."""
    
    def __init__(self):
        """Initialize price extractor with patterns."""
        # Crypto currencies
        self.crypto_symbols = ['BTC', 'ETH', 'USDT', 'XMR', 'LTC', 'BCH', 'XRP', 'DOGE', 'DASH', 'ZEC']
        
        # Fiat currencies
        self.fiat_symbols = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'RUB', 'INR']
        self.fiat_signs = ['$', '€', '£', '¥', '₹', '₽']
        
        # Build comprehensive patterns
        self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns for price extraction."""
        # Pattern for numbers with optional commas and decimals
        number_pattern = r'\d+(?:,\d{3})*(?:\.\d{1,2})?'
        
        # Crypto pattern: number + crypto symbol (e.g., "15 BTC", "100ETH")
        crypto_pattern = rf'({number_pattern})\s*({"|".join(self.crypto_symbols)})'
        
        # Fiat with symbol before: $1,000, €500, £1000
        fiat_prefix_pattern = rf'([{"".join(self.fiat_signs)}])\s*({number_pattern})'
        
        # Fiat with currency code after: 1000 USD, 500 EUR
        fiat_suffix_pattern = rf'({number_pattern})\s*({"|".join(self.fiat_symbols)})'
        
        # Fiat with symbol and code: $1000 USD
        fiat_full_pattern = rf'([{"".join(self.fiat_signs)}])\s*({number_pattern})\s*({"|".join(self.fiat_symbols)})?'
        
        # Combined pattern
        self.patterns = [
            (re.compile(crypto_pattern, re.IGNORECASE), 'crypto'),
            (re.compile(fiat_prefix_pattern), 'fiat_prefix'),
            (re.compile(fiat_suffix_pattern, re.IGNORECASE), 'fiat_suffix'),
            (re.compile(fiat_full_pattern), 'fiat_full')
        ]
    
    def extract_prices(self, text: str) -> List[Dict[str, any]]:
        """
        Extract all prices from text.
        
        Args:
            text: Input text to search
            
        Returns:
            List of dictionaries with 'amount', 'currency', 'type' (crypto/fiat)
        """
        prices = []
        
        if not text:
            return prices
        
        for pattern, pattern_type in self.patterns:
            matches = pattern.finditer(text)
            for match in matches:
                if pattern_type == 'crypto':
                    amount_str = match.group(1)
                    currency = match.group(2).upper()
                    amount = self._parse_amount(amount_str)
                    prices.append({
                        'amount': amount,
                        'currency': currency,
                        'type': 'crypto',
                        'original': match.group(0)
                    })
                
                elif pattern_type == 'fiat_prefix':
                    currency_sign = match.group(1)
                    amount_str = match.group(2)
                    amount = self._parse_amount(amount_str)
                    currency = self._sign_to_currency(currency_sign)
                    prices.append({
                        'amount': amount,
                        'currency': currency,
                        'type': 'fiat',
                        'original': match.group(0)
                    })
                
                elif pattern_type == 'fiat_suffix':
                    amount_str = match.group(1)
                    currency = match.group(2).upper()
                    amount = self._parse_amount(amount_str)
                    prices.append({
                        'amount': amount,
                        'currency': currency,
                        'type': 'fiat',
                        'original': match.group(0)
                    })
                
                elif pattern_type == 'fiat_full':
                    currency_sign = match.group(1)
                    amount_str = match.group(2)
                    currency_code = match.group(3)
                    amount = self._parse_amount(amount_str)
                    currency = currency_code.upper() if currency_code else self._sign_to_currency(currency_sign)
                    prices.append({
                        'amount': amount,
                        'currency': currency,
                        'type': 'fiat',
                        'original': match.group(0)
                    })
        
        # Remove duplicates (same amount and currency)
        seen = set()
        unique_prices = []
        for price in prices:
            key = (price['amount'], price['currency'])
            if key not in seen:
                seen.add(key)
                unique_prices.append(price)
        
        return unique_prices
    
    def extract_first_price(self, text: str) -> Optional[Dict[str, any]]:
        """
        Extract the first price found in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with 'amount' and 'currency', or None
        """
        prices = self.extract_prices(text)
        return prices[0] if prices else None
    
    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string to float."""
        # Remove commas
        cleaned = amount_str.replace(',', '')
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    def _sign_to_currency(self, sign: str) -> str:
        """Convert currency sign to currency code."""
        mapping = {
            '$': 'USD',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '₹': 'INR',
            '₽': 'RUB'
        }
        return mapping.get(sign, 'USD')

