"""
sorting_algorithms.py - Sorting Algorithm Step Generators
===========================================================
Each function is a Python *generator* that yields one SortStep per
atomic operation (comparison or swap). This lets the GUI drive the
animation at any speed by simply calling next() on the generator.

Why generators?
  - Zero threads needed for animation — the GUI timer calls next().
  - The algorithm logic reads exactly like the textbook pseudocode.
  - Easy to pause/resume: just stop calling next().
"""

from typing import Generator, List
from utils import SortStep, BarState


# ── Bubble Sort ─────────────────────────────────────────────────────────────

def bubble_sort(arr: List[int]) -> Generator[SortStep, None, None]:
    """
    Bubble Sort — O(n²) average/worst, O(n) best (with early exit)
    ----------------------------------------------------------------
    Idea: In each pass, adjacent pairs are compared and swapped if out
    of order. The largest unsorted element 'bubbles' to the end.
    
    Optimization: if no swaps happen in a full pass, the array is sorted.
    """
    a = arr[:]
    n = len(a)

    for i in range(n):
        swapped = False

        for j in range(0, n - i - 1):
            # Highlight the pair being compared
            yield SortStep(
                array=a[:],
                highlights={j: BarState.COMPARE, j + 1: BarState.COMPARE,
                            **{k: BarState.SORTED for k in range(n - i, n)}},
                message=f"Comparing a[{j}]={a[j]} and a[{j+1}]={a[j+1]}"
            )

            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
                # Highlight the swap
                yield SortStep(
                    array=a[:],
                    highlights={j: BarState.SWAP, j + 1: BarState.SWAP,
                                **{k: BarState.SORTED for k in range(n - i, n)}},
                    message=f"Swapped! a[{j}]={a[j]} ↔ a[{j+1}]={a[j+1]}"
                )

        # Mark the element that just settled at the end as sorted
        yield SortStep(
            array=a[:],
            highlights={**{k: BarState.SORTED for k in range(n - i - 1, n)}},
            message=f"Pass {i+1} complete — a[{n-i-1}] is now in place"
        )

        if not swapped:
            break   # Early exit: array is already sorted

    # Final state — all sorted
    yield SortStep(
        array=a[:],
        highlights={k: BarState.SORTED for k in range(n)},
        message="✓ Array is fully sorted!"
    )


# ── Merge Sort ──────────────────────────────────────────────────────────────

def merge_sort(arr: List[int]) -> Generator[SortStep, None, None]:
    """
    Merge Sort — O(n log n) guaranteed, O(n) auxiliary space, Stable
    -----------------------------------------------------------------
    Divide: split array in halves recursively until size 1.
    Conquer: merge pairs of sorted sub-arrays into a larger sorted array.
    
    We use an iterative bottom-up approach here so we can yield steps
    without stack overflow issues on large arrays.
    """
    a = arr[:]
    n = len(a)
    sorted_indices: set = set()

    # Bottom-up merge sort: merge sub-arrays of size 1, then 2, then 4 …
    width = 1
    while width < n:
        for lo in range(0, n, 2 * width):
            mid   = min(lo + width,      n)
            hi    = min(lo + 2 * width,  n)

            left  = a[lo:mid]
            right = a[mid:hi]

            i = j = 0
            k = lo

            while i < len(left) and j < len(right):
                yield SortStep(
                    array=a[:],
                    highlights={lo + i: BarState.COMPARE, mid + j: BarState.COMPARE},
                    message=f"Comparing left[{i}]={left[i]} and right[{j}]={right[j]}"
                )
                if left[i] <= right[j]:
                    a[k] = left[i];  i += 1
                else:
                    a[k] = right[j]; j += 1
                yield SortStep(
                    array=a[:],
                    highlights={k: BarState.SWAP},
                    message=f"Placed {a[k]} at index {k}"
                )
                k += 1

            # Copy remaining elements
            while i < len(left):
                a[k] = left[i]; i += 1; k += 1
                yield SortStep(array=a[:], highlights={k-1: BarState.SWAP},
                               message=f"Copying remaining left element {a[k-1]}")
            while j < len(right):
                a[k] = right[j]; j += 1; k += 1
                yield SortStep(array=a[:], highlights={k-1: BarState.SWAP},
                               message=f"Copying remaining right element {a[k-1]}")

        width *= 2

    yield SortStep(
        array=a[:],
        highlights={k: BarState.SORTED for k in range(n)},
        message="✓ Array is fully sorted!"
    )


# ── Quick Sort ──────────────────────────────────────────────────────────────

def quick_sort(arr: List[int]) -> Generator[SortStep, None, None]:
    """
    Quick Sort — O(n log n) average, O(n²) worst, O(log n) stack space
    -------------------------------------------------------------------
    Pick a pivot (last element), partition so all elements < pivot are
    to the left, then recursively sort each partition.
    
    We use an explicit stack instead of recursion so we can yield steps.
    Pivot choice: last element (simple; median-of-3 would be better in prod).
    """
    a = arr[:]
    n = len(a)
    sorted_indices: set = set()

    # Iterative quick sort using an explicit stack of (lo, hi) bounds
    stack = [(0, n - 1)]

    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            sorted_indices.add(lo)
            continue

        # ── Partition around pivot = a[hi] ──────────────────────────
        pivot = a[hi]
        yield SortStep(
            array=a[:],
            highlights={hi: BarState.PIVOT, **{k: BarState.SORTED for k in sorted_indices}},
            message=f"Pivot = {pivot} at index {hi}"
        )

        store = lo    # store index for elements < pivot

        for j in range(lo, hi):
            yield SortStep(
                array=a[:],
                highlights={j: BarState.COMPARE, hi: BarState.PIVOT,
                            **{k: BarState.SORTED for k in sorted_indices}},
                message=f"Comparing a[{j}]={a[j]} with pivot={pivot}"
            )

            if a[j] <= pivot:
                a[store], a[j] = a[j], a[store]
                yield SortStep(
                    array=a[:],
                    highlights={store: BarState.SWAP, j: BarState.SWAP,
                                hi: BarState.PIVOT,
                                **{k: BarState.SORTED for k in sorted_indices}},
                    message=f"a[{j}]={a[j]} ≤ pivot — swap with store index {store}"
                )
                store += 1

        # Place pivot in its final sorted position
        a[store], a[hi] = a[hi], a[store]
        sorted_indices.add(store)
        yield SortStep(
            array=a[:],
            highlights={store: BarState.SORTED,
                        **{k: BarState.SORTED for k in sorted_indices}},
            message=f"Pivot {pivot} placed at final index {store}"
        )

        # Push sub-problems onto stack
        stack.append((lo, store - 1))
        stack.append((store + 1, hi))

    yield SortStep(
        array=a[:],
        highlights={k: BarState.SORTED for k in range(n)},
        message="✓ Array is fully sorted!"
    )


# ── Heap Sort ───────────────────────────────────────────────────────────────

def heap_sort(arr: List[int]) -> Generator[SortStep, None, None]:
    """
    Heap Sort — O(n log n) guaranteed, O(1) auxiliary space, Not stable
    -------------------------------------------------------------------
    Phase 1 (heapify): Build a max-heap from the array — O(n).
    Phase 2 (extract): Repeatedly swap the root (max) with the last
                       unsorted element, shrink the heap, and sift down.
    
    A max-heap satisfies: parent ≥ both children for every node.
    """
    a = arr[:]
    n = len(a)

    def sift_down(arr, root, end, sorted_set, phase_msg=""):
        """Restore the max-heap property by sifting the root down."""
        while True:
            largest = root
            left    = 2 * root + 1
            right   = 2 * root + 2

            if left < end and arr[left] > arr[largest]:
                largest = left
            if right < end and arr[right] > arr[largest]:
                largest = right

            if largest == root:
                break

            yield SortStep(
                array=arr[:],
                highlights={root: BarState.COMPARE, largest: BarState.COMPARE,
                            **{k: BarState.SORTED for k in sorted_set}},
                message=f"{phase_msg} Sifting down: swap a[{root}]={arr[root]} ↔ a[{largest}]={arr[largest]}"
            )

            arr[root], arr[largest] = arr[largest], arr[root]
            yield SortStep(
                array=arr[:],
                highlights={root: BarState.SWAP, largest: BarState.SWAP,
                            **{k: BarState.SORTED for k in sorted_set}},
                message=f"{phase_msg} Swapped — heap property restored locally"
            )
            root = largest

    sorted_set: set = set()

    # ── Phase 1: Build max-heap ──────────────────────────────────────
    # Start from the last non-leaf and sift down every node
    for i in range(n // 2 - 1, -1, -1):
        yield from sift_down(a, i, n, sorted_set, phase_msg="[Build heap]")

    yield SortStep(
        array=a[:],
        highlights={0: BarState.PIVOT},
        message="Max-heap built — root holds the maximum element"
    )

    # ── Phase 2: Extract max elements one by one ─────────────────────
    for end in range(n - 1, 0, -1):
        # Swap root (max) with last unsorted element
        a[0], a[end] = a[end], a[0]
        sorted_set.add(end)

        yield SortStep(
            array=a[:],
            highlights={0: BarState.SWAP, end: BarState.SORTED,
                        **{k: BarState.SORTED for k in sorted_set}},
            message=f"Extracted max={a[end]} → placed at index {end}"
        )

        # Restore heap property for the reduced heap
        yield from sift_down(a, 0, end, sorted_set, phase_msg="[Extract]")

    sorted_set.add(0)
    yield SortStep(
        array=a[:],
        highlights={k: BarState.SORTED for k in range(n)},
        message="✓ Array is fully sorted!"
    )
