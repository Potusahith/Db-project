#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <limits.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int compare_ulonglong(const void *a, const void *b) {
    unsigned long long arg1 = *(const unsigned long long*)a;
    unsigned long long arg2 = *(const unsigned long long*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

typedef struct {
    double mean;
    unsigned long long median;
    unsigned long long mode;
    unsigned long long min;
    unsigned long long max;
    unsigned long long p25;
    unsigned long long p75;
} Statistics;

Statistics calculate_statistics(unsigned long long *data, long long size, int num_threads) {
    Statistics stats;
    omp_set_num_threads(num_threads);
    
    unsigned long long min_val = ULLONG_MAX;
    unsigned long long max_val = 0;
    double sum = 0.0;
    
    #pragma omp parallel
    {
        unsigned long long local_min = ULLONG_MAX;
        unsigned long long local_max = 0;
        double local_sum = 0.0;
        
        #pragma omp for
        for (long long i = 0; i < size; i++) {
            if (data[i] < local_min) local_min = data[i];
            if (data[i] > local_max) local_max = data[i];
            local_sum += data[i];
        }
        
        #pragma omp critical
        {
            if (local_min < min_val) min_val = local_min;
            if (local_max > max_val) max_val = local_max;
            sum += local_sum;
        }
    }
    
    stats.min = min_val;
    stats.max = max_val;
    stats.mean = sum / size;
    
    qsort(data, size, sizeof(unsigned long long), compare_ulonglong);
    
    if (size % 2 == 0) {
        stats.median = (data[size/2 - 1] + data[size/2]) / 2;
    } else {
        stats.median = data[size/2];
    }
    
    stats.p25 = data[(long long)(size * 0.25)];
    stats.p75 = data[(long long)(size * 0.75)];
    stats.mode = data[size/2];
    
    return stats;
}

void save_sample_data(unsigned long long *data, long long size, const char *filename, long long sample_size) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return;
    
    long long to_save = (sample_size < size) ? sample_size : size;
    for (long long i = 0; i < to_save; i++) {
        fprintf(fp, "%llu\n", data[i]);
    }
    fclose(fp);
}

// Scenario A: 100,000 values/second × 3,600 seconds = 360,000,000 values
double problem5a_streaming_data(int num_threads, int save_data) {
    omp_set_num_threads(num_threads);
    
    long long total_values = 360000000LL;  // 100K/sec × 3600 sec
    
    unsigned long long *data = malloc(total_values * sizeof(unsigned long long));
    if (!data) {
        fprintf(stderr, "Memory allocation failed for Problem 5a\n");
        return -1;
    }
    
    double start_time = get_time();
    
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num() * 42 + 12345;
        
        #pragma omp for
        for (long long i = 0; i < total_values; i++) {
            unsigned long long val1 = rand_r(&seed);
            unsigned long long val2 = rand_r(&seed);
            data[i] = (val1 << 32) | val2;
            data[i] %= 1000000000000ULL;
        }
    }
    
    Statistics stats = calculate_statistics(data, total_values, num_threads);
    
    double end_time = get_time();
    double execution_time = end_time - start_time;
    
    printf("Threads: %2d | Mean: %.2e | Median: %llu | Min: %llu | Max: %llu | Time: %.4f s\n", 
           num_threads, stats.mean, stats.median, stats.min, stats.max, execution_time);
    
    if (save_data) {
        save_sample_data(data, total_values, "problem5a_data.txt", 100000);
        printf("           | Saved sample to problem5a_data.txt\n");
    }
    
    free(data);
    return execution_time;
}

// Scenario B: 60,000,000 values/minute × 60 minutes = 3,600,000,000 values
// Using 100M sample for memory constraints
double problem5b_streaming_data(int num_threads, int save_data) {
    omp_set_num_threads(num_threads);
    
    // Note: Full size would be 3.6B, using 100M sample
    long long sample_size = 100000000LL;
    
    unsigned long long *data = malloc(sample_size * sizeof(unsigned long long));
    if (!data) {
        fprintf(stderr, "Memory allocation failed for Problem 5b\n");
        return -1;
    }
    
    double start_time = get_time();
    
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num() * 43 + 54321;
        
        #pragma omp for
        for (long long i = 0; i < sample_size; i++) {
            unsigned long long val1 = rand_r(&seed);
            unsigned long long val2 = rand_r(&seed);
            data[i] = (val1 << 32) | val2;
            data[i] %= 1000000000000ULL;
        }
    }
    
    Statistics stats = calculate_statistics(data, sample_size, num_threads);
    
    double end_time = get_time();
    double execution_time = end_time - start_time;
    
    printf("Threads: %2d | Mean: %.2e | Median: %llu | Min: %llu | Max: %llu | Time: %.4f s\n", 
           num_threads, stats.mean, stats.median, stats.min, stats.max, execution_time);
    
    if (save_data) {
        save_sample_data(data, sample_size, "problem5b_data.txt", 100000);
        printf("           | Saved sample to problem5b_data.txt\n");
    }
    
    free(data);
    return execution_time;
}

int main() {
    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12, 14, 16};
    int num_configs = 9;
    int runs = 5;
    
    printf("=================================================================\n");
    printf("PROBLEM 5: Streaming Data Analysis\n");
    printf("=================================================================\n\n");
    
    double results_a[9];
    double results_b[9];
    
    // Scenario A
    printf("SCENARIO A: 100,000 values/second for 1 hour (360M values)\n");
    printf("------------------------------------------------------------\n");
    for (int i = 0; i < num_configs; i++) {
        int threads = thread_counts[i];
        double total_time = 0.0;
        
        printf("\nRunning with %d thread(s) - %d iterations:\n", threads, runs);
        
        for (int run = 0; run < runs; run++) {
            printf("  Run %d: ", run + 1);
            int save_data = (i == 0 && run == 0) ? 1 : 0;
            double exec_time = problem5a_streaming_data(threads, save_data);
            total_time += exec_time;
        }
        
        results_a[i] = total_time / runs;
        printf("  Average time: %.4f seconds\n", results_a[i]);
    }
    
    // Scenario B
    printf("\n\n=================================================================\n");
    printf("SCENARIO B: 60M values/minute for 1 hour (3.6B values, using 100M sample)\n");
    printf("------------------------------------------------------------\n");
    for (int i = 0; i < num_configs; i++) {
        int threads = thread_counts[i];
        double total_time = 0.0;
        
        printf("\nRunning with %d thread(s) - %d iterations:\n", threads, runs);
        
        for (int run = 0; run < runs; run++) {
            printf("  Run %d: ", run + 1);
            int save_data = (i == 0 && run == 0) ? 1 : 0;
            double exec_time = problem5b_streaming_data(threads, save_data);
            total_time += exec_time;
        }
        
        results_b[i] = total_time / runs;
        printf("  Average time: %.4f seconds\n", results_b[i]);
    }
    
    // Speedup Analysis
    printf("\n\n=================================================================\n");
    printf("SPEEDUP ANALYSIS - Scenario A\n");
    printf("=================================================================\n");
    printf("Threads | Time (s)  | Speedup | Efficiency\n");
    printf("--------|-----------|---------|------------\n");
    
    double baseline_a = results_a[0];
    for (int i = 0; i < num_configs; i++) {
        double speedup = baseline_a / results_a[i];
        double efficiency = (speedup / thread_counts[i]) * 100.0;
        printf("  %2d    | %9.4f | %7.2f | %7.2f%%\n", 
               thread_counts[i], results_a[i], speedup, efficiency);
    }
    
    printf("\n=================================================================\n");
    printf("SPEEDUP ANALYSIS - Scenario B\n");
    printf("=================================================================\n");
    printf("Threads | Time (s)  | Speedup | Efficiency\n");
    printf("--------|-----------|---------|------------\n");
    
    double baseline_b = results_b[0];
    for (int i = 0; i < num_configs; i++) {
        double speedup = baseline_b / results_b[i];
        double efficiency = (speedup / thread_counts[i]) * 100.0;
        printf("  %2d    | %9.4f | %7.2f | %7.2f%%\n", 
               thread_counts[i], results_b[i], speedup, efficiency);
    }
    
    // Save results
    FILE *fp = fopen("problem5_results.txt", "w");
    fprintf(fp, "Threads,ScenarioA_Time(s),ScenarioA_Speedup,ScenarioB_Time(s),ScenarioB_Speedup\n");
    for (int i = 0; i < num_configs; i++) {
        fprintf(fp, "%d,%.4f,%.2f,%.4f,%.2f\n", 
                thread_counts[i], 
                results_a[i], baseline_a / results_a[i],
                results_b[i], baseline_b / results_b[i]);
    }
    fclose(fp);
    
    printf("\n=================================================================\n");
    printf("Results saved to problem5_results.txt\n");
    printf("Data samples: problem5a_data.txt, problem5b_data.txt\n");
    printf("\nNext step: Run 'python3 problem5_visualize.py' for box plots\n");
    printf("=================================================================\n");
    
    return 0;
}