#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <cmath>
#include <algorithm>
#include "secondary_index_engine.h"

enum class DistributionType {
    UNIFORM,
    POISSON,
    SKEWED_NORMAL,
    TEMPORAL_HOTSPOT
};

struct BenchmarkConfig {
    size_t total_records;
    std::string db_path;
    std::string leveldb_path;
    size_t flush_threshold;
    DistributionType distribution;
    size_t primary_key_space;
    size_t secondary_key_space;
    
    double poisson_lambda;
    double skewed_mean;
    double skewed_std;
    double skewed_skewness;
    double hotspot_ratio;
};

struct BenchmarkResult {
    double index_write_qps;
    double index_avg_latency;
    double index_write_time;
    size_t index_records_written;

    double leveldb_write_qps;
    double leveldb_avg_latency;
    double leveldb_write_time;
    size_t leveldb_records_written;
};

class DataGenerator {
public:
    DataGenerator(const BenchmarkConfig& config) 
        : config_(config), gen_(std::random_device{}()) {
        initializeDistributions();
    }

    std::pair<uint32_t, uint32_t> generateKeyValue() {
        uint32_t primary_key = generatePrimaryKey();
        uint32_t secondary_key = generateSecondaryKey();
        return {primary_key, secondary_key};
    }

private:
    void initializeDistributions() {
        primary_uniform_ = std::uniform_int_distribution<uint32_t>(0, config_.primary_key_space - 1);
        secondary_uniform_ = std::uniform_int_distribution<uint32_t>(0, config_.secondary_key_space - 1);
        
        poisson_lambda_ = config_.poisson_lambda;
        primary_poisson_ = std::poisson_distribution<uint32_t>(poisson_lambda_);
        secondary_poisson_ = std::poisson_distribution<uint32_t>(poisson_lambda_);
        
        skewed_mean_ = config_.skewed_mean;
        skewed_std_ = config_.skewed_std;
        skewed_skewness_ = config_.skewed_skewness;
        
        hotspot_ratio_ = config_.hotspot_ratio;
        hotspot_size_ = static_cast<size_t>(config_.secondary_key_space * hotspot_ratio_);
        hotspot_center_ = config_.secondary_key_space / 2;
    }

    uint32_t generatePrimaryKey() {
        switch (config_.distribution) {
            case DistributionType::UNIFORM:
                return primary_uniform_(gen_);
            case DistributionType::POISSON:
                return std::min(primary_poisson_(gen_), static_cast<uint32_t>(config_.primary_key_space - 1));
            case DistributionType::SKEWED_NORMAL:
                return generateSkewedNormalKey(config_.primary_key_space);
            case DistributionType::TEMPORAL_HOTSPOT:
                return generateTemporalHotspotKey(config_.primary_key_space);
            default:
                return primary_uniform_(gen_);
        }
    }

    uint32_t generateSecondaryKey() {
        switch (config_.distribution) {
            case DistributionType::UNIFORM:
                return secondary_uniform_(gen_);
            case DistributionType::POISSON:
                return std::min(secondary_poisson_(gen_), static_cast<uint32_t>(config_.secondary_key_space - 1));
            case DistributionType::SKEWED_NORMAL:
                return generateSkewedNormalKey(config_.secondary_key_space);
            case DistributionType::TEMPORAL_HOTSPOT:
                return generateTemporalHotspotKey(config_.secondary_key_space);
            default:
                return secondary_uniform_(gen_);
        }
    }

    uint32_t generateSkewedNormalKey(size_t key_space) {
        double u1 = std::uniform_real_distribution<double>(0.0, 1.0)(gen_);
        double u2 = std::uniform_real_distribution<double>(0.0, 1.0)(gen_);
        
        double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        double z1 = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);
        
        double skewed_z = z0 + skewed_skewness_ * (z0 * z0 - 1.0) / 6.0;
        
        double normalized_value = skewed_z * skewed_std_ + skewed_mean_;
        uint32_t key = static_cast<uint32_t>(std::max(0.0, std::min(static_cast<double>(key_space - 1), normalized_value)));
        return key;
    }

    uint32_t generateTemporalHotspotKey(size_t key_space) {
        double u = std::uniform_real_distribution<double>(0.0, 1.0)(gen_);
        
        if (u < hotspot_ratio_) {
            std::uniform_int_distribution<uint32_t> hotspot_dist(
                std::max(0u, static_cast<uint32_t>(hotspot_center_ - hotspot_size_ / 2)),
                std::min(static_cast<uint32_t>(key_space - 1), 
                        static_cast<uint32_t>(hotspot_center_ + hotspot_size_ / 2))
            );
            return hotspot_dist(gen_);
        } else {
            std::uniform_int_distribution<uint32_t> non_hotspot_dist(0, static_cast<uint32_t>(key_space - 1));
            uint32_t key;
            do {
                key = non_hotspot_dist(gen_);
            } while (key >= std::max(0u, static_cast<uint32_t>(hotspot_center_ - hotspot_size_ / 2)) &&
                     key <= std::min(static_cast<uint32_t>(key_space - 1), 
                                   static_cast<uint32_t>(hotspot_center_ + hotspot_size_ / 2)));
            return key;
        }
    }

    BenchmarkConfig config_;
    std::mt19937 gen_;
    
    std::uniform_int_distribution<uint32_t> primary_uniform_;
    std::uniform_int_distribution<uint32_t> secondary_uniform_;
    
    double poisson_lambda_;
    std::poisson_distribution<uint32_t> primary_poisson_;
    std::poisson_distribution<uint32_t> secondary_poisson_;
    
    double skewed_mean_;
    double skewed_std_;
    double skewed_skewness_;
    
    double hotspot_ratio_;
    size_t hotspot_size_;
    size_t hotspot_center_;
};

class InsertBenchmark {
public:
    InsertBenchmark(const BenchmarkConfig& config) 
        : config_(config), engine_(nullptr), data_gen_(config) {
    }

    BenchmarkResult run() {
        BenchmarkResult result{};
        
        printTestConfig();
        
        std::cout << "Starting secondary index engine insert benchmark...\n";
        runSecondaryIndexTest(result);

        std::cout << "\nStarting LevelDB insert benchmark...\n";
        runLevelDBTest(result);

        printResults(result);

        return result;
    }

private:
    void printTestConfig() {
        std::cout << "============== Benchmark Configuration ==============\n";
        std::cout << "Total records: " << config_.total_records << "\n";
        std::cout << "Primary key space: " << config_.primary_key_space << "\n";
        std::cout << "Secondary key space: " << config_.secondary_key_space << "\n";
        std::cout << "Flush threshold: " << config_.flush_threshold << "\n\n";
        
        std::cout << "Data distribution type: ";
        
        switch (config_.distribution) {
            case DistributionType::UNIFORM:
                std::cout << "Uniform\n";
                break;
            case DistributionType::POISSON:
                std::cout << "Poisson (lambda=" << config_.poisson_lambda << ")\n";
                break;
            case DistributionType::SKEWED_NORMAL:
                std::cout << "Skewed Normal (mean=" << config_.skewed_mean 
                         << ", std=" << config_.skewed_std 
                         << ", skewness=" << config_.skewed_skewness << ")\n";
                break;
            case DistributionType::TEMPORAL_HOTSPOT:
                std::cout << "Temporal Hotspot (hotspot ratio=" << config_.hotspot_ratio << ")\n";
                break;
        }
        std::cout << "====================================================\n\n";
    }

    void runSecondaryIndexTest(BenchmarkResult& result) {
        engine_ = std::make_unique<SecondaryIndexEngine>(
            config_.db_path, "Temperature", config_.flush_threshold);

        auto write_start = std::chrono::high_resolution_clock::now();
        result.index_records_written = runIndexWriteTest();
        auto write_end = std::chrono::high_resolution_clock::now();
        
        auto write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            write_end - write_start).count();
        result.index_write_time = write_duration;
        result.index_write_qps = result.index_records_written * 1000.0 / write_duration;
        
        if (!write_latencies_.empty()) {
            double total_latency = 0;
            for (const auto& latency : write_latencies_) {
                total_latency += latency;
            }
            result.index_avg_latency = total_latency / write_latencies_.size();
        }
    }

    void runLevelDBTest(BenchmarkResult& result) {
        leveldb::DB* db;
        leveldb::Options options;
        options.create_if_missing = true;
        leveldb::Status status = leveldb::DB::Open(options, config_.leveldb_path, &db);
        
        if (!status.ok()) {
            std::cerr << "无法打开LevelDB数据库: " << status.ToString() << std::endl;
            return;
        }

        std::unique_ptr<leveldb::DB> db_guard(db);
        std::vector<double> leveldb_latencies;
        result.leveldb_records_written = 0;

        auto write_start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < config_.total_records; ++i) {
            auto [primary_key, secondary_key] = data_gen_.generateKeyValue();
            
            std::string key = std::to_string(primary_key);
            std::string value = std::to_string(secondary_key);
            
            auto op_start = std::chrono::high_resolution_clock::now();
            
            status = db->Put(leveldb::WriteOptions(), key, value);
            
            auto op_end = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration_cast<std::chrono::microseconds>(
                op_end - op_start).count() / 1000.0;
            leveldb_latencies.push_back(latency);

            if (status.ok()) {
                result.leveldb_records_written++;
            }

            if ((i + 1) % (config_.total_records / 10) == 0) {
                double progress = (i + 1) * 100.0 / config_.total_records;
                std::cout << "LevelDB write progress: " << std::fixed << std::setprecision(1) 
                         << progress << "% (" << (i + 1) << "/" 
                         << config_.total_records << ")\n";
            }
        }

        auto write_end = std::chrono::high_resolution_clock::now();
        
        auto write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            write_end - write_start).count();
        result.leveldb_write_time = write_duration;
        result.leveldb_write_qps = result.leveldb_records_written * 1000.0 / write_duration;

        if (!leveldb_latencies.empty()) {
            double total_latency = 0;
            for (const auto& latency : leveldb_latencies) {
                total_latency += latency;
            }
            result.leveldb_avg_latency = total_latency / leveldb_latencies.size();
        }
    }

    size_t runIndexWriteTest() {
        size_t records_written = 0;
        const size_t progress_interval = config_.total_records / 10;

        for (size_t i = 0; i < config_.total_records; ++i) {
            if (insertRandomRecord()) {
                records_written++;
            }
            
            if (progress_interval > 0 && (i + 1) % progress_interval == 0) {
                double progress = (i + 1) * 100.0 / config_.total_records;
                std::cout << "Secondary index write progress: " << std::fixed << std::setprecision(1) 
                         << progress << "% (" << (i + 1) << "/" 
                         << config_.total_records << ")\n";
            }
        }

        std::cout << "Write finished, total records written: " << records_written << "\n";
        return records_written;
    }

    bool insertRandomRecord() {
        auto [primary_key, secondary_key] = data_gen_.generateKeyValue();

        auto start = std::chrono::high_resolution_clock::now();
        bool success = engine_->insert_temperature(primary_key, secondary_key);
        auto end = std::chrono::high_resolution_clock::now();

        double latency = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count() / 1000.0;
        write_latencies_.push_back(latency);

        return success;
    }

    void printResults(const BenchmarkResult& result) {
        std::cout << "============== Insert Benchmark Results ==============\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Scale:\n";
        std::cout << "  Total records:       " << config_.total_records << "\n";
        std::cout << "  Primary key space:   " << config_.primary_key_space << "\n";
        std::cout << "  Secondary key space: " << config_.secondary_key_space << "\n";
        std::cout << "  Flush threshold:     " << config_.flush_threshold << "\n\n";
        
        std::cout << "Secondary Index Engine Performance:\n";
        std::cout << "  Total write time:    " << result.index_write_time << " ms\n";
        std::cout << "  Write throughput:    " << result.index_write_qps << " ops/s\n";
        std::cout << "  Avg write latency:   " << result.index_avg_latency << " ms\n";
        std::cout << "  Records written:     " << result.index_records_written << "\n\n";

        std::cout << "LevelDB Baseline Performance:\n";
        std::cout << "  Total write time:    " << result.leveldb_write_time << " ms\n";
        std::cout << "  Write throughput:    " << result.leveldb_write_qps << " ops/s\n";
        std::cout << "  Avg write latency:   " << result.leveldb_avg_latency << " ms\n";
        std::cout << "  Records written:     " << result.leveldb_records_written << "\n\n";

        std::cout << "Performance Comparison (Secondary Index / LevelDB):\n";
        double write_qps_ratio = result.index_write_qps / result.leveldb_write_qps;
        double latency_ratio = result.index_avg_latency / result.leveldb_avg_latency;
        double time_ratio = result.index_write_time / result.leveldb_write_time;
        std::cout << "  Write time ratio:    " << time_ratio << "x\n";
        std::cout << "  Throughput ratio:    " << write_qps_ratio << "x\n";
        std::cout << "  Latency ratio:       " << latency_ratio << "x\n\n";
        
        if (!write_latencies_.empty()) {
            std::vector<double> sorted_latencies = write_latencies_;
            std::sort(sorted_latencies.begin(), sorted_latencies.end());
            
            std::cout << "Secondary index write latency distribution:\n";
            std::cout << "  P50 latency:         " << sorted_latencies[sorted_latencies.size() * 0.5] << " ms\n";
            std::cout << "  P90 latency:         " << sorted_latencies[sorted_latencies.size() * 0.9] << " ms\n";
            std::cout << "  P99 latency:         " << sorted_latencies[sorted_latencies.size() * 0.99] << " ms\n";
        }
        
        std::cout << "====================================================\n\n";
    }

    BenchmarkConfig config_;
    std::unique_ptr<SecondaryIndexEngine> engine_;
    DataGenerator data_gen_;
    std::vector<double> write_latencies_;
};

int main() {
    DistributionType distribution = DistributionType::UNIFORM;
    
    BenchmarkConfig config{
        .total_records = 10000,
        .db_path = "/opt/Leveldb_DB_DOC/leveldb_benchmark",
        .leveldb_path = "/opt/Leveldb_DB_DOC/leveldb_baseline",
        .flush_threshold = 100000,
        .distribution = distribution,
        .primary_key_space = 100000000,
        .secondary_key_space = 4000000,
        
        .poisson_lambda = 1000000,
        .skewed_mean = 2000000,
        .skewed_std = 800000,
        .skewed_skewness = 2.0,
        .hotspot_ratio = 0.1,
    };

    InsertBenchmark benchmark(config);
    benchmark.run();

    return 0;
}