"""
Timer utility for performance monitoring and benchmarking.

This module provides a simple Timer class for measuring execution time of different
code sections and operations. It allows marking specific points in time and provides
detailed timing reports for performance analysis and optimization.

The Timer class maintains a chronological list of timing marks and can generate
both real-time output and summary reports showing the duration of each marked
section and total execution time.

Example:
    >>> timer = Timer()
    >>> timer.mark("data_loading")
    >>> # ... data loading code ...
    >>> timer.mark("processing")
    >>> # ... processing code ...
    >>> timer.print_times()
"""

from time import time

class Timer:
    """
    A utility class for measuring and reporting execution times.
    
    This class provides functionality to mark specific points during program execution
    and measure the time elapsed between these marks. It's useful for performance
    profiling, benchmarking, and identifying bottlenecks in code execution.
    
    The timer automatically starts when instantiated and maintains a chronological
    list of all timing marks. Each mark records both a descriptive tag and the
    timestamp when it was created.
    
    Attributes:
        times_list (List[Tuple[str, float]]): List of (tag, timestamp) tuples
                                            recording all timing marks in chronological order.
                                            Automatically initialized with ("start", current_time).
    
    Example:
        >>> timer = Timer()
        >>> timer.mark("initialization")
        initialization took 0.05 seconds
        >>> timer.mark("computation")
        computation took 2.30 seconds
        >>> timer.print_times()
        Timing results:
        initialization: 0.05 seconds
        computation: 2.30 seconds
        Total time: 2.35 seconds
    """
    
    def __init__(self):
        """
        Initialize the Timer with the current timestamp as the start time.
        
        Creates a new Timer instance and automatically records the initialization
        time as the starting point for all subsequent timing measurements.
        The start time is stored with the tag "start" and can be used as a
        reference point for calculating total execution time.
        
        Side Effects:
            - Initializes times_list with the current timestamp
            - Records the start time for subsequent duration calculations
            
        Example:
            >>> timer = Timer()
            >>> # Timer is now ready to mark execution points
        """
        self.times_list=[("start", time())]

    def mark(self, tag: str) -> None:
        """
        Mark a specific point in time with a descriptive tag.
        
        Records the current timestamp with the provided tag and calculates
        the elapsed time since the previous mark. Immediately prints the
        duration of the most recent interval for real-time feedback.
        
        This method is useful for tracking progress through different phases
        of execution and identifying time-consuming operations.
        
        Args:
            tag (str): Descriptive label for this timing mark.
                      Should be meaningful and describe what operation
                      just completed or is about to begin.
        
        Side Effects:
            - Appends (tag, timestamp) to times_list
            - Prints elapsed time since previous mark to stdout
            
        Example:
            >>> timer = Timer()
            >>> timer.mark("data_loading")
            data_loading took 1.23 seconds
            >>> timer.mark("preprocessing")
            preprocessing took 0.45 seconds
        """
        t=time()
        prev_t = self.times_list[-1][1] if self.times_list else 0
        self.times_list.append((tag, t))
        print(f"{tag} took {t - prev_t:.2f} seconds")

    def print_times(self) -> None:
        """
        Print a comprehensive timing report showing all marked intervals.
        
        Generates and displays a detailed summary of all timing marks,
        showing the duration of each marked interval and the total elapsed
        time since initialization. The report excludes the initial "start"
        mark from the detailed breakdown but uses it to calculate total time.
        
        The output format shows:
        - Individual interval durations with their tags
        - Total execution time from start to last mark
        - All times formatted to 2 decimal places
        
        Side Effects:
            - Prints formatted timing report to stdout
            
        Example Output:
            Timing results:
            initialization: 0.05 seconds
            data_loading: 1.23 seconds
            processing: 2.30 seconds
            cleanup: 0.12 seconds
            Total time: 3.70 seconds
            
        Example:
            >>> timer = Timer()
            >>> timer.mark("phase1")
            >>> timer.mark("phase2")
            >>> timer.print_times()
            Timing results:
            phase1: 0.15 seconds
            phase2: 0.32 seconds
            Total time: 0.47 seconds
        """
        print("Timing results:")
        for i, (tag, t) in enumerate(self.times_list):
            if tag == "start":
                continue
            prev_t = self.times_list[i - 1][1]
            print(f"{tag}: {t - prev_t:.2f} seconds")
        total_time = self.times_list[-1][1] - self.times_list[0][1]
        print(f"Total time: {total_time:.2f} seconds")


# ------------------- Main / Example Usage ---------------------
if __name__ == "__main__":
    """
    Demonstration of Timer class functionality.
    
    This example shows how to use the Timer class to measure execution
    time of different operations. It simulates computational work with
    increasing complexity and demonstrates both real-time feedback
    and summary reporting capabilities.
    
    The example creates progressively more work in each step to show
    how timing differences are captured and reported.
    """
    timer = Timer()
    
    # Simulate some computational work with increasing complexity
    print("Starting timing demonstration...")
    
    for i in range(5):
        # Simulate work that takes progressively longer
        for j in range(i*1000000):
            pass
        timer.mark(f"step {i+1}")
    
    print("\n" + "="*50)
    timer.print_times()
    print("="*50)
    print("Timer demonstration complete!")