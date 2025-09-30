#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

#define NUM_SUBSEQUENCES 1000
#define ELEMENTS_PER_SEQ 1000000
#define TOTAL_ELEMENTS (NUM_SUBSEQUENCES * ELEMENTS_PER_SEQ)

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int compare_ints(const void *a, const void *b) {
    int arg1 = *(const int*)a;
    int arg2 = *(const int*)b;
    return (arg1 > arg2) - (arg1 < arg2);
}

double problem3_sorting_merging(int num_threads) {  // Return time
    omp_set_num_threads(num_threads);
    
    int *data = malloc(TOTAL_ELEMENTS * sizeof(int));
    if (!data) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }
    
    double start_time = get_time();
    
    // Generate data
    #pragma omp parallel for
    for (int seq = 0; seq < NUM_SUBSEQUENCES; seq++) {
        unsigned int seed = seq * 42 + 12345;
        int base_value = seq * 1000;
        
        for (int i = 0; i < ELEMENTS_PER_SEQ; i++) {
            data[seq * ELEMENTS_PER_SEQ + i] = base_value + (rand_r(&seed) % 1000);
        }
    }
    
    // Parallel sort
    #pragma omp parallel for
    for (int seq = 0; seq < NUM_SUBSEQUENCES; seq++) {
        qsort(&data[seq * ELEMENTS_PER_SEQ], ELEMENTS_PER_SEQ, sizeof(int), compare_ints);
    }
    
    double end_time = get_time();
    double execution_time = end_time - start_time;
    
    printf("Threads: %2d | Elements: %d | Time: %.4f s\n", 
           num_threads, TOTAL_ELEMENTS, execution_time);
    
    free(data);
    return execution_time;
}

int main() {
    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12, 14, 16};
    int num_configs = 9;
    int runs = 5;
    
    printf("=================================================================\n");
    printf("PROBLEM 3: Sorting and Merging Subsequences\n");
    printf("=================================================================\n\n");
    
    double results[9];
    
    for (int i = 0; i < num_configs; i++) {
        int threads = thread_counts[i];
        double total_time = 0.0;
        
        printf("Running with %d thread(s) - %d iterations:\n", threads, runs);
        
        for (int run = 0; run < runs; run++) {
            printf("  Run %d: ", run + 1);
            double exec_time = problem3_sorting_merging(threads);
            total_time += exec_time;
        }
        
        results[i] = total_time / runs;
        printf("  Average time: %.4f seconds\n\n", results[i]);
    }
    
    // Print speedup table
    printf("\n=================================================================\n");
    printf("SPEEDUP ANALYSIS\n");
    printf("=================================================================\n");
    printf("Threads | Time (s)  | Speedup | Efficiency\n");
    printf("--------|-----------|---------|------------\n");
    
    double baseline = results[0];
    for (int i = 0; i < num_configs; i++) {
        double speedup = baseline / results[i];
        double efficiency = (speedup / thread_counts[i]) * 100.0;
        printf("  %2d    | %9.4f | %7.2f | %7.2f%%\n", 
               thread_counts[i], results[i], speedup, efficiency);
    }
    
    // Save results
    FILE *fp = fopen("problem3_results.txt", "w");
    fprintf(fp, "Threads,Time(s),Speedup,Efficiency(%%)\n");
    for (int i = 0; i < num_configs; i++) {
        double speedup = baseline / results[i];
        double efficiency = (speedup / thread_counts[i]) * 100.0;
        fprintf(fp, "%d,%.4f,%.2f,%.2f\n", 
                thread_counts[i], results[i], speedup, efficiency);
    }
    fclose(fp);
    printf("\nResults saved to problem3_results.txt\n");
    
    return 0;
}