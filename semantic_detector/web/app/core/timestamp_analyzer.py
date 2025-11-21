"""Timestamp analysis and response time calculation."""

from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dateutil import parser as date_parser
import numpy as np


class TimestampAnalyzer:
    """Analyze timestamps and calculate response times."""
    
    def __init__(self):
        """Initialize timestamp analyzer."""
        pass
    
    def parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """
        Parse timestamp string to datetime object.
        
        Args:
            timestamp_str: Timestamp string in various formats
            
        Returns:
            datetime object or None if parsing fails
        """
        if not timestamp_str:
            return None
        
        try:
            # Try dateutil parser (handles most formats)
            return date_parser.parse(timestamp_str)
        except (ValueError, TypeError):
            # Try common formats manually
            formats = [
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            return None
    
    def calculate_response_time(
        self, 
        timestamp1: Optional[datetime], 
        timestamp2: Optional[datetime]
    ) -> Optional[float]:
        """
        Calculate response time in minutes between two timestamps.
        
        Args:
            timestamp1: First timestamp
            timestamp2: Second timestamp
            
        Returns:
            Response time in minutes, or None if calculation not possible
        """
        if not timestamp1 or not timestamp2:
            return None
        
        try:
            delta = timestamp2 - timestamp1
            return delta.total_seconds() / 60.0  # Convert to minutes
        except Exception:
            return None
    
    def calculate_response_times(self, messages: List[Dict]) -> List[Optional[float]]:
        """
        Calculate response times for a list of messages.
        
        Args:
            messages: List of message dicts with 'timestamp' field
            
        Returns:
            List of response times in minutes (None for first message or missing timestamps)
        """
        response_times = []
        
        if not messages:
            return response_times
        
        # First message has no response time
        response_times.append(None)
        
        for i in range(1, len(messages)):
            prev_msg = messages[i - 1]
            curr_msg = messages[i]
            
            prev_ts = self.parse_timestamp(prev_msg.get('timestamp'))
            curr_ts = self.parse_timestamp(curr_msg.get('timestamp'))
            
            response_time = self.calculate_response_time(prev_ts, curr_ts)
            response_times.append(response_time)
        
        return response_times
    
    def get_response_time_stats(self, response_times: List[Optional[float]]) -> Dict[str, float]:
        """
        Calculate statistics for response times.
        
        Args:
            response_times: List of response times in minutes
            
        Returns:
            Dictionary with statistics
        """
        valid_times = [rt for rt in response_times if rt is not None]
        
        if not valid_times:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        return {
            'count': len(valid_times),
            'mean': float(np.mean(valid_times)),
            'median': float(np.median(valid_times)),
            'std': float(np.std(valid_times)),
            'min': float(np.min(valid_times)),
            'max': float(np.max(valid_times))
        }
    
    def create_histogram_bins(self, response_times: List[Optional[float]], bin_type: str = 'auto') -> Tuple[List[float], List[int]]:
        """
        Create histogram bins for response time distribution.
        
        Args:
            response_times: List of response times in minutes
            bin_type: 'minutes', 'hours', 'days', or 'auto'
            
        Returns:
            Tuple of (bin_edges, counts)
        """
        valid_times = [rt for rt in response_times if rt is not None]
        
        if not valid_times:
            return ([], [])
        
        max_time = max(valid_times)
        
        # Determine appropriate binning
        if bin_type == 'auto':
            if max_time < 60:  # Less than 1 hour
                bin_type = 'minutes'
            elif max_time < 1440:  # Less than 1 day
                bin_type = 'hours'
            else:
                bin_type = 'days'
        
        if bin_type == 'minutes':
            bins = np.arange(0, max_time + 60, 60)  # 1-hour bins
        elif bin_type == 'hours':
            bins = np.arange(0, max_time + 1440, 1440)  # 1-day bins
        elif bin_type == 'days':
            bins = np.arange(0, max_time + 1440 * 7, 1440 * 7)  # 1-week bins
        else:
            # Default: 30 bins
            bins = np.linspace(0, max_time, 31)
        
        counts, bin_edges = np.histogram(valid_times, bins=bins)
        
        return (bin_edges.tolist(), counts.tolist())

