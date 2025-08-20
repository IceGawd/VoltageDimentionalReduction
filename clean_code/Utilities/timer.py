"""
Timer utility for code performance profiling and execution time tracking.

This module provides a simple but effective timing utility to measure execution time
between different points in the code. It's particularly useful for:
- Performance profiling
- Code optimization
- Execution time reporting
- Process monitoring

The Timer class maintains a chronological list of timestamps with associated tags,
allowing for both real-time reporting and summary statistics of execution times.
"""

from time import time

class Timer:
    """
    A utility class for measuring and reporting execution times.
    
    The Timer automatically starts when initialized and can mark multiple timing points
    throughout code execution. Each timing point is tagged for identification and
    the time differences between consecutive points are tracked.
    
    Attributes:
        times_list (list): List of tuples containing (tag, timestamp) pairs.
            The first entry is always ("start", start_time).
            
    Example:
        >>> timer = Timer()
        >>> # Do some work
        >>> timer.mark("data loading")
        >>> # Do more work
        >>> timer.mark("processing")
        >>> timer.print_times()  # Prints all timing results
    """
    
    def __init__(self):
        """
        Initialize a new Timer instance.
        
        Creates a new timer and records the start time automatically with
        the tag "start".
        """
        self.times_list=[("start", time())]

    def mark(self, tag: str) -> None:
        """
        Mark a timing point with a given tag and print the elapsed time.
        
        Records the current time and calculates the elapsed time since the previous
        mark. The timing information is both stored for later retrieval and
        immediately printed.
        
        Args:
            tag (str): Identifier for this timing point. Should be descriptive
                of the code section that was timed.
                
        Note:
            The elapsed time is calculated from the previous mark, not from
            the start. For total time from start, use print_times().
            
        Example:
            >>> timer = Timer()
            >>> # Some time-consuming operation
            >>> timer.mark("database query")  # Prints time taken for query
        """
        t=time()
        prev_t = self.times_list[-1][1] if self.times_list else 0
        self.times_list.append((tag, t))
        print(f"{tag} took {t - prev_t:.2f} seconds")

    def print_times(self) -> None:
        """
        Print a summary of all timing results.
        
        Displays:
        - Individual timing results for each marked section
        - Total execution time from start to last mark
        
        The start time is not included in the individual results, but is used
        to calculate the total execution time.
        
        Format:
            Timing results:
            <tag1>: <time1> seconds
            <tag2>: <time2> seconds
            ...
            Total time: <total> seconds
            
        Example:
            >>> timer = Timer()
            >>> # Multiple operations with timer.mark()
            >>> timer.print_times()
            Timing results:
            data loading: 1.23 seconds
            processing: 0.85 seconds
            Total time: 2.08 seconds
        """
        print("Timing results:")
        for i, (tag, t) in enumerate(self.times_list):
            if tag == "start":
                continue
            prev_t = self.times_list[i - 1][1]
            print(f"{tag}: {t - prev_t:.2f} seconds")
        total_time = self.times_list[-1][1] - self.times_list[0][1]
        print(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    timer = Timer()
    timer.mark("start")
    # Simulate some work
    for i in range(5):
        for j in range(i*1000000):
            pass
        timer.mark(f"step {i+1}")
    timer.print_times()