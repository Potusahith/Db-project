#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <limits.h>

#define N (1LL << 34)  // 2^34 elements
#define DOMAIN_MAX 1000000000  // 10^9

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

double problem1_min_max_mean(int num_threads) {  // Return execution time
    omp_set_num_threads(num_threads);
    
    long long min_val = LLONG_MAX;
    long long max_val = LLONG_MIN;
    double sum = 0.0;
    
    double start_time = get_time();
    
    #pragma omp parallel
    {
        long long local_min = LLONG_MAX;
        long long local_max = LLONG_MIN;
        double local_sum = 0.0;
        unsigned int seed = omp_get_thread_num() * 42 + 12345;
        
        #pragma omp for
        for (long long i = 0; i < N; i++) {
            long long value = rand_r(&seed) % (DOMAIN_MAX + 1);
            
            if (value < local_min) local_min = value;
            if (value > local_max) local_max = value;
            local_sum += value;
        }
        
        #pragma omp critical
        {
            if (local_min < min_val) min_val = local_min;
            if (local_max > max_val) max_val = local_max;
            sum += local_sum;
        }
    }
    
    double end_time = get_time();
    double execution_time = end_time - start_time;
    double mean = sum / N;
    
    printf("Threads: %2d | Min: %lld | Max: %lld | Mean: %.2f | Time: %.4f s\n", 
           num_threads, min_val, max_val, mean, execution_time);
    
    return execution_time;  // FIXED: Return the time
}

int main() {
    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12, 14, 16};
    int num_configs = 9;
    int runs = 5;
    
    printf("=================================================================\n");
    printf("PROBLEM 1: Minimum, Maximum, and Mean (2^34 elements)\n");
    printf("=================================================================\n\n");
    
    double results[9];
    
    for (int i = 0; i < num_configs; i++) {
        int threads = thread_counts[i];
        double total_time = 0.0;
        
        printf("Running with %d thread(s) - %d iterations:\n", threads, runs);
        
        for (int run = 0; run < runs; run++) {
            printf("  Run %d: ", run + 1);
            double exec_time = problem1_min_max_mean(threads);  // FIXED: Capture return value
            total_time += exec_time;  // FIXED: Accumulate time
        }
        
        results[i] = total_time / runs;  // FIXED: Calculate average
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
    
    // Save results to file
    FILE *fp = fopen("problem1_results.txt", "w");
    fprintf(fp, "Threads,Time(s),Speedup,Efficiency(%%)\n");
    for (int i = 0; i < num_configs; i++) {
        double speedup = baseline / results[i];
        double efficiency = (speedup / thread_counts[i]) * 100.0;
        fprintf(fp, "%d,%.4f,%.2f,%.2f\n", 
                thread_counts[i], results[i], speedup, efficiency);
    }
    fclose(fp);
    printf("\nResults saved to problem1_results.txt\n");
    
    return 0;
}