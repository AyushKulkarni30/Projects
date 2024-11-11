import random
import time

def quick_sort_deterministic(arr, low, high):
    if low < high:
        pi = partition_deterministic(arr, low, high)
        quick_sort_deterministic(arr, low, pi - 1)
        quick_sort_deterministic(arr, pi + 1, high)

def partition_deterministic(arr, low, high):
    pivot = arr[high]  # Choosing the last element as the pivot
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quick_sort_randomized(arr, low, high):
    if low < high:
        pi = partition_randomized(arr, low, high)
        quick_sort_randomized(arr, low, pi - 1)
        quick_sort_randomized(arr, pi + 1, high)

def partition_randomized(arr, low, high):
    pivot_index = random.randint(low, high)  # Choosing a random pivot
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def analyze_quick_sort(arr):
    def measure_time(sort_func, arr):
        arr_copy = arr.copy()
        start_time = time.time()
        sort_func(arr_copy, 0, len(arr_copy) - 1)
        end_time = time.time()
        return end_time - start_time

    print("Array size:", len(arr))

    deterministic_time = measure_time(quick_sort_deterministic, arr)
    print(f"Deterministic Quick Sort Time: {deterministic_time:.6f} seconds")

    randomized_time = measure_time(quick_sort_randomized, arr)
    print(f"Randomized Quick Sort Time: {randomized_time:.6f} seconds")

if __name__ == "__main__":
    # Test cases with different array sizes
    sizes = [100, 1000, 5000]

    for size in sizes:
        arr = [random.randint(0, 10000) for _ in range(size)]
        print("\nTesting with array of size", size)
        analyze_quick_sort(arr)


