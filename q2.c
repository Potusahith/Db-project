#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

#define N 1000000000LL  // 10^9 elements

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

double problem2_dot_product_reduction(int num_threads) {  // Return time
    omp_set_num_threads(num_threads);
    
    long long dot_product = 0;
    
    double start_time = get_time();
    
    #pragma omp parallel reduction(+:dot_product)
    {
        unsigned int seed = omp_get_thread_num() * 42 + 12345;
        
        #pragma omp for
        for (long long i = 0; i < N; i++) {
            int a = (rand_r(&seed) % 3) - 1;
            int b = (rand_r(&seed) % 3) - 1;
            dot_product += (long long)a * b;
        }
    }
    
    double end_time = get_time();
    double execution_time = end_time - start_time;
    
    printf("Threads: %2d | Dot Product: %lld | Time: %.4f s\n", 
           num_threads, dot_product, execution_time);
    
    return execution_time;
}

int main() {
    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12, 14, 16};
    int num_configs = 9;
    int runs = 5;
    
    printf("=================================================================\n");
    printf("PROBLEM 2: Dot Product (10^9 elements from {-1, 0, 1})\n");
    printf("=================================================================\n\n");
    
    double results[9];
    
    for (int i = 0; i < num_configs; i++) {
        int threads = thread_counts[i];
        double total_time = 0.0;
        
        printf("Running with %d thread(s) - %d iterations:\n", threads, runs);
        
        for (int run = 0; run < runs; run++) {
            printf("  Run %d: ", run + 1);
            double exec_time = problem2_dot_product_reduction(threads);
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
    FILE *fp = fopen("problem2_results.txt", "w");
    fprintf(fp, "Threads,Time(s),Speedup,Efficiency(%%)\n");
    for (int i = 0; i < num_configs; i++) {
        double speedup = baseline / results[i];
        double efficiency = (speedup / thread_counts[i]) * 100.0;
        fprintf(fp, "%d,%.4f,%.2f,%.2f\n", 
                thread_counts[i], results[i], speedup, efficiency);
    }
    fclose(fp);
    printf("\nResults saved to problem2_results.txt\n");
    
    return 0;
}