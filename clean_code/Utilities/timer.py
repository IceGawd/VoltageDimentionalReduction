from time import time

class Timer:
    def __init__(self):
        self.times_list=[("start", time())]

    def mark(self,tag):
        t=time()
        prev_t = self.times_list[-1][1] if self.times_list else 0
        self.times_list.append((tag, t))
        print(f"{tag} took {t - prev_t:.2f} seconds")

    def print_times(self):
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