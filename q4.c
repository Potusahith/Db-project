#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <string.h>

#define MATRIX_SIZE 4096

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

double matrix_multiply_block(int n, int block_size, int num_threads) {
    omp_set_num_threads(num_threads);
    
    // Allocate matrices
    double **A = malloc(n * sizeof(double*));
    double **B = malloc(n * sizeof(double*));
    double **C = malloc(n * sizeof(double*));
    
    for (int i = 0; i < n; i++) {
        A[i] = malloc(n * sizeof(double));
        B[i] = malloc(n * sizeof(double));
        C[i] = malloc(n * sizeof(double));
    }
    
    // Initialize matrices
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)rand_r(&seed) / RAND_MAX;
            B[i][j] = (double)rand_r(&seed) / RAND_MAX;
            C[i][j] = 0.0;
        }
    }
    
    double start_time = get_time();
    
    // Block matrix multiplication
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                // Compute block
                int i_max = (ii + block_size < n) ? ii + block_size : n;
                int j_max = (jj + block_size < n) ? jj + block_size : n;
                int k_max = (kk + block_size < n) ? kk + block_size : n;
                
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        double a_ik = A[i][k];
                        for (int j = jj; j < j_max; j++) {
                            C[i][j] += a_ik * B[k][j];
                        }
                    }
                }
            }
        }
    }
    
    double end_time = get_time();
    double execution_time = end_time - start_time;
    
    // Free memory
    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
    
    return execution_time;
}

int main() {
    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12, 14, 16};
    int block_sizes[] = {2, 4, 8, 16, 32};
    int num_thread_configs = 9;
    int num_block_configs = 5;
    int runs = 5;
    
    printf("=================================================================\n");
    printf("PROBLEM 4: Block Matrix Multiplication (%dx%d)\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("=================================================================\n\n");
    
    // Results: [thread_config][block_size]
    double results[9][5];
    
    // Test each block size
    for (int bs = 0; bs < num_block_configs; bs++) {
        int block_size = block_sizes[bs];
        
        printf("\n=================================================================\n");
        printf("BLOCK SIZE: %d\n", block_size);
        printf("=================================================================\n");
        
        for (int tc = 0; tc < num_thread_configs; tc++) {
            int threads = thread_counts[tc];
            double total_time = 0.0;
            
            printf("Running with %d thread(s) - %d iterations:\n", threads, runs);
            
            for (int run = 0; run < runs; run++) {
                printf("  Run %d: ", run + 1);
                double exec_time = matrix_multiply_block(MATRIX_SIZE, block_size, threads);
                printf("Time: %.4f s\n", exec_time);
                total_time += exec_time;
            }
            
            results[tc][bs] = total_time / runs;
            printf("  Average time: %.4f seconds\n\n", results[tc][bs]);
        }
    }
    
    // Print speedup analysis for each block size
    printf("\n=================================================================\n");
    printf("SPEEDUP ANALYSIS BY BLOCK SIZE\n");
    printf("=================================================================\n\n");
    
    for (int bs = 0; bs < num_block_configs; bs++) {
        int block_size = block_sizes[bs];
        double baseline = results[0][bs];
        
        printf("Block Size %d:\n", block_size);
        printf("Threads | Time (s)  | Speedup | Efficiency\n");
        printf("--------|-----------|---------|------------\n");
        
        for (int tc = 0; tc < num_thread_configs; tc++) {
            double speedup = baseline / results[tc][bs];
            double efficiency = (speedup / thread_counts[tc]) * 100.0;
            printf("  %2d    | %9.4f | %7.2f | %7.2f%%\n", 
                   thread_counts[tc], results[tc][bs], speedup, efficiency);
        }
        printf("\n");
    }
    
    // Comparison table across block sizes
    printf("\n=================================================================\n");
    printf("SPEEDUP COMPARISON (All Block Sizes)\n");
    printf("=================================================================\n");
    printf("Threads | Block2 | Block4 | Block8 | Block16 | Block32\n");
    printf("--------|--------|--------|--------|---------|--------\n");
    
    for (int tc = 0; tc < num_thread_configs; tc++) {
        printf("  %2d    |", thread_counts[tc]);
        for (int bs = 0; bs < num_block_configs; bs++) {
            double speedup = results[0][bs] / results[tc][bs];
            printf(" %6.2f |", speedup);
        }
        printf("\n");
    }
    
    // Save detailed results to CSV
    FILE *fp = fopen("problem4_results.txt", "w");
    fprintf(fp, "Threads");
    for (int bs = 0; bs < num_block_configs; bs++) {
        fprintf(fp, ",Block%d_Time,Block%d_Speedup", block_sizes[bs], block_sizes[bs]);
    }
    fprintf(fp, "\n");
    
    for (int tc = 0; tc < num_thread_configs; tc++) {
        fprintf(fp, "%d", thread_counts[tc]);
        for (int bs = 0; bs < num_block_configs; bs++) {
            double speedup = results[0][bs] / results[tc][bs];
            fprintf(fp, ",%.4f,%.2f", results[tc][bs], speedup);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("\nResults saved to problem4_results.txt\n");
    
    // Find optimal configuration
    printf("\n=================================================================\n");
    printf("OPTIMAL CONFIGURATIONS\n");
    printf("=================================================================\n");
    
    for (int tc = 0; tc < num_thread_configs; tc++) {
        double min_time = results[tc][0];
        int best_block = block_sizes[0];
        
        for (int bs = 1; bs < num_block_configs; bs++) {
            if (results[tc][bs] < min_time) {
                min_time = results[tc][bs];
                best_block = block_sizes[bs];
            }
        }
        
        printf("Threads %2d: Best block size = %2d (Time: %.4f s)\n", 
               thread_counts[tc], best_block, min_time);
    }
    
    return 0;
}