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
    size_t query_count;
    std::string db_path;
    std::string leveldb_path;
    size_t flush_threshold;
    uint32_t range_width;
    bool print_results;
    DistributionType distribution;
    size_t primary_key_space;
    size_t secondary_key_space;
    
    double poisson_lambda;
    double skewed_mean;
    double skewed_std;
    double skewed_skewness;
    double hotspot_ratio;
};

struct QueryBenchmarkResult {
    double index_point_qps;
    double index_point_latency;
    double index_range_qps;
    double index_range_latency;
    
    double leveldb_point_qps;
    double leveldb_point_latency;
    double leveldb_range_qps;
    double leveldb_range_latency;
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

class QueryBenchmark {
public:
    QueryBenchmark(const BenchmarkConfig& config) 
        : config_(config), engine_(nullptr), data_gen_(config) {
        generateQueryParameters();
    }

    QueryBenchmarkResult run() {
        QueryBenchmarkResult result{};
        
        printTestConfig();
        
        std::cout << "Starting to insert test data...\n";
        insertTestData();

        std::cout << "\nStarting secondary index engine query benchmark...\n";
        runSecondaryIndexQueries(result);

        std::cout << "\nStarting LevelDB scan query benchmark...\n";
        runLevelDBQueries(result);

        printResults(result);

        return result;
    }

private:
    void printTestConfig() {
        std::cout << "============== Benchmark Configuration ==============" << std::endl;
        std::cout << "Total records: " << config_.total_records << std::endl;
        std::cout << "Query count: " << config_.query_count << std::endl;
        std::cout << "Primary key space: " << config_.primary_key_space << std::endl;
        std::cout << "Secondary key space: " << config_.secondary_key_space << std::endl;
        std::cout << "Flush threshold: " << config_.flush_threshold << std::endl;
        std::cout << "Range query width: " << config_.range_width << std::endl;
        std::cout << "Data distribution type: ";
        
        switch (config_.distribution) {
            case DistributionType::UNIFORM:
                std::cout << "Uniform" << std::endl;
                break;
            case DistributionType::POISSON:
                std::cout << "Poisson (lambda=" << config_.poisson_lambda << ")" << std::endl;
                break;
            case DistributionType::SKEWED_NORMAL:
                std::cout << "Skewed Normal (mean=" << config_.skewed_mean 
                          << ", std=" << config_.skewed_std 
                          << ", skewness=" << config_.skewed_skewness << ")" << std::endl;
                break;
            case DistributionType::TEMPORAL_HOTSPOT:
                std::cout << "Temporal Hotspot (hotspot ratio=" << config_.hotspot_ratio << ")" << std::endl;
                break;
        }
        std::cout << "====================================================\n" << std::endl;
    }

    void insertTestData() {
        engine_ = std::make_unique<SecondaryIndexEngine>(
            config_.db_path, "Temperature", config_.flush_threshold);

        leveldb::DB* db;
        leveldb::Options options;
        options.create_if_missing = true;
        leveldb::Status status = leveldb::DB::Open(options, config_.leveldb_path, &db);
        if (!status.ok()) {
            throw std::runtime_error("Failed to open LevelDB database: " + status.ToString());
        }
        std::unique_ptr<leveldb::DB> db_guard(db);

        std::cout << "Starting to insert data...\n";
        for (size_t i = 0; i < config_.total_records; ++i) {
            auto [primary_key, secondary_key] = data_gen_.generateKeyValue();

            engine_->insert_temperature(primary_key, secondary_key);

            std::string db_key = std::to_string(primary_key);
            std::string value = std::to_string(secondary_key);
            db->Put(leveldb::WriteOptions(), db_key, value);

            if ((i + 1) % (config_.total_records / 10) == 0) {
                std::cout << "Insert progress: " << (i + 1) * 100.0 / config_.total_records 
                         << "% (" << (i + 1) << "/" << config_.total_records << ")\n";
            }
        }
        std::cout << "Data insertion completed\n";
    }

    void generateQueryParameters() {
        point_query_temps_.reserve(config_.query_count);
        for (size_t i = 0; i < config_.query_count; ++i) {
            auto [_, secondary_key] = data_gen_.generateKeyValue();
            point_query_temps_.push_back(secondary_key);
        }

        range_query_bounds_.reserve(config_.query_count);
        for (size_t i = 0; i < config_.query_count; ++i) {
            auto [_, secondary_key] = data_gen_.generateKeyValue();
            uint32_t lower = secondary_key;
            uint32_t upper = std::min(static_cast<uint32_t>(config_.secondary_key_space - 1), 
                                    lower + config_.range_width);
            range_query_bounds_.push_back({lower, upper});
        }
    }

    void runSecondaryIndexQueries(QueryBenchmarkResult& result) {
        std::vector<double> point_latencies;
        std::vector<double> range_latencies;
        
        std::cout << "\nStarting secondary index point query test...\n";
        auto point_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.query_count; ++i) {
            uint32_t target_temp = point_query_temps_[i];
            
            auto query_start = std::chrono::high_resolution_clock::now();
            auto records = engine_->query_by_attribute(target_temp);
            auto query_end = std::chrono::high_resolution_clock::now();
            
            double latency = std::chrono::duration_cast<std::chrono::microseconds>(
                query_end - query_start).count() / 1000.0;
            point_latencies.push_back(latency);

            if (config_.print_results) {
                std::cout << "\nQuery temperature value: " << target_temp << "\n";
                std::cout << "Found records: " << records.size() << "\n";
                if (!records.empty()) {
                    std::cout << "Record details:\n";
                    for (const auto& record : records) {
                        std::cout << "  Primary key: " << record.primary_key 
                                << ", Temperature: " << record.attributes.at("Temperature") << "\n";
                    }
                }
                std::cout << "Query latency: " << latency << " ms\n";
                std::cout << "----------------------------------------\n";
            }
        }
        auto point_end = std::chrono::high_resolution_clock::now();
        
        std::cout << "\nStarting secondary index range query test...\n";
        auto range_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.query_count; ++i) {
            const auto& [lower, upper] = range_query_bounds_[i];
            
            auto query_start = std::chrono::high_resolution_clock::now();
            auto records = engine_->range_query_by_attribute(lower, upper);
            auto query_end = std::chrono::high_resolution_clock::now();
            
            double latency = std::chrono::duration_cast<std::chrono::microseconds>(
                query_end - query_start).count() / 1000.0;
            range_latencies.push_back(latency);

            if (config_.print_results) {
                std::cout << "\nRange query: [" << lower << ", " << upper << "]\n";
                std::cout << "Found records: " << records.size() << "\n";
                if (!records.empty()) {
                    std::cout << "Record details (showing first 5):\n";
                    for (size_t j = 0; j < std::min(size_t(5), records.size()); ++j) {
                        std::cout << "  Primary key: " << records[j].primary_key 
                                << ", Temperature: " << records[j].attributes.at("Temperature") << "\n";
                    }
                    if (records.size() > 5) {
                        std::cout << "  ... and " << records.size() - 5 << " more records ...\n";
                    }
                }
                std::cout << "Query latency: " << latency << " ms\n";
                std::cout << "----------------------------------------\n";
            }
        }
        auto range_end = std::chrono::high_resolution_clock::now();

        calculateQueryMetrics(point_start, point_end, point_latencies,
                            result.index_point_qps, result.index_point_latency);
        calculateQueryMetrics(range_start, range_end, range_latencies,
                            result.index_range_qps, result.index_range_latency);
    }

    void runLevelDBQueries(QueryBenchmarkResult& result) {
        leveldb::DB* db;
        leveldb::Options options;
        options.create_if_missing = true;
        leveldb::Status status = leveldb::DB::Open(options, config_.leveldb_path, &db);
        if (!status.ok()) {
            throw std::runtime_error("Failed to open LevelDB database: " + status.ToString());
        }
        std::unique_ptr<leveldb::DB> db_guard(db);

        std::vector<double> point_latencies;
        std::vector<double> range_latencies;
        
        std::cout << "\nStarting LevelDB point query test...\n";
        auto point_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.query_count; ++i) {
            uint32_t target_temp = point_query_temps_[i];
            
            auto query_start = std::chrono::high_resolution_clock::now();
            std::vector<uint32_t> results;
            
            std::unique_ptr<leveldb::Iterator> it(db->NewIterator(leveldb::ReadOptions()));
            for (it->SeekToFirst(); it->Valid(); it->Next()) {
                uint32_t temp = std::stoul(it->value().ToString());
                if (temp == target_temp) {
                    results.push_back(std::stoul(it->key().ToString()));
                }
            }
            
            auto query_end = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration_cast<std::chrono::microseconds>(
                query_end - query_start).count() / 1000.0;
            point_latencies.push_back(latency);

            if (config_.print_results) {
                std::cout << "\nQuery temperature value: " << target_temp << "\n";
                std::cout << "Found records: " << results.size() << "\n";
                if (!results.empty()) {
                    std::cout << "Record details:\n";
                    for (const auto& key : results) {
                        std::cout << "  Primary key: " << key << ", Temperature: " << target_temp << "\n";
                    }
                }
                std::cout << "Query latency: " << latency << " ms\n";
                std::cout << "----------------------------------------\n";
            }
        }
        auto point_end = std::chrono::high_resolution_clock::now();
        
        std::cout << "\nStarting LevelDB range query test...\n";
        auto range_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.query_count; ++i) {
            const auto& [lower, upper] = range_query_bounds_[i];
            
            auto query_start = std::chrono::high_resolution_clock::now();
            std::vector<uint32_t> results;
            
            std::unique_ptr<leveldb::Iterator> it(db->NewIterator(leveldb::ReadOptions()));
            for (it->SeekToFirst(); it->Valid(); it->Next()) {
                uint32_t temp = std::stoul(it->value().ToString());
                if (temp >= lower && temp <= upper) {
                    results.push_back(std::stoul(it->key().ToString()));
                }
            }
            
            auto query_end = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration_cast<std::chrono::microseconds>(
                query_end - query_start).count() / 1000.0;
            range_latencies.push_back(latency);

            if (config_.print_results) {
                std::cout << "\nRange query: [" << lower << ", " << upper << "]\n";
                std::cout << "Found records: " << results.size() << "\n";
                if (!results.empty()) {
                    std::cout << "Record details (showing first 5):\n";
                    for (size_t j = 0; j < std::min(size_t(5), results.size()); ++j) {
                        std::string value;
                        std::string key = std::to_string(results[j]);
                        status = db->Get(leveldb::ReadOptions(), key, &value);
                        if (status.ok()) {
                            uint32_t temp = std::stoul(value);
                            std::cout << "  Primary key: " << results[j] 
                                    << ", Temperature: " << temp << "\n";
                        }
                    }
                    if (results.size() > 5) {
                        std::cout << "  ... and " << results.size() - 5 << " more records ...\n";
                    }
                }
                std::cout << "Query latency: " << latency << " ms\n";
                std::cout << "----------------------------------------\n";
            }
        }
        auto range_end = std::chrono::high_resolution_clock::now();

        calculateQueryMetrics(point_start, point_end, point_latencies,
                            result.leveldb_point_qps, result.leveldb_point_latency);
        calculateQueryMetrics(range_start, range_end, range_latencies,
                            result.leveldb_range_qps, result.leveldb_range_latency);
    }

    void calculateQueryMetrics(
        const std::chrono::high_resolution_clock::time_point& start,
        const std::chrono::high_resolution_clock::time_point& end,
        const std::vector<double>& latencies,
        double& qps,
        double& avg_latency) {
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start).count();
        
        qps = config_.query_count * 1000.0 / duration;
        
        avg_latency = 0;
        for (const auto& latency : latencies) {
            avg_latency += latency;
        }
        avg_latency /= latencies.size();
    }

    void printResults(const QueryBenchmarkResult& result) {
        std::cout << "\n============== Query Performance Test Results ==============\n";
        std::cout << std::fixed << std::setprecision(2);
        
        std::cout << "Test Scale:\n";
        std::cout << "  Total records:     " << config_.total_records << "\n";
        std::cout << "  Query count:       " << config_.query_count << "\n";
        std::cout << "  Primary key space: " << config_.primary_key_space << "\n";
        std::cout << "  Secondary key space: " << config_.secondary_key_space << "\n";
        std::cout << "  Range width:       " << config_.range_width << "\n\n";
        
        std::cout << "Secondary Index Engine Performance:\n";
        std::cout << "  Point query QPS:   " << result.index_point_qps << " ops/s\n";
        std::cout << "  Point query latency: " << result.index_point_latency << " ms\n";
        std::cout << "  Range query QPS:   " << result.index_range_qps << " ops/s\n";
        std::cout << "  Range query latency: " << result.index_range_latency << " ms\n\n";
        
        std::cout << "LevelDB Scan Query Performance:\n";
        std::cout << "  Point query QPS:   " << result.leveldb_point_qps << " ops/s\n";
        std::cout << "  Point query latency: " << result.leveldb_point_latency << " ms\n";
        std::cout << "  Range query QPS:   " << result.leveldb_range_qps << " ops/s\n";
        std::cout << "  Range query latency: " << result.leveldb_range_latency << " ms\n\n";
        
        std::cout << "Performance Comparison (Secondary Index/LevelDB):\n";
        std::cout << "  Point query QPS ratio: " << result.index_point_qps / result.leveldb_point_qps << "x\n";
        std::cout << "  Point query latency ratio: " << result.index_point_latency / result.leveldb_point_latency << "x\n";
        std::cout << "  Range query QPS ratio: " << result.index_range_qps / result.leveldb_range_qps << "x\n";
        std::cout << "  Range query latency ratio: " << result.index_range_latency / result.leveldb_range_latency << "x\n";
        
        std::cout << "==========================================\n\n";
    }

    BenchmarkConfig config_;
    std::unique_ptr<SecondaryIndexEngine> engine_;
    DataGenerator data_gen_;
    
    std::vector<uint32_t> point_query_temps_;
    std::vector<std::pair<uint32_t, uint32_t>> range_query_bounds_;
};

int main() {
    DistributionType distribution = DistributionType::UNIFORM;
    
    BenchmarkConfig config{
        .total_records = 1000000,
        .query_count = 100,
        .db_path = "/opt/Leveldb_DB_DOC/leveldb_benchmark",
        .leveldb_path = "/opt/Leveldb_DB_DOC/leveldb_baseline",
        .flush_threshold = 100000,
        .range_width = 1000,
        .print_results = false,
        .distribution = distribution,
        .primary_key_space = 100000000,
        .secondary_key_space = 4000000,
        
        .poisson_lambda = 1000000,
        .skewed_mean = 2000000,
        .skewed_std = 800000,
        .skewed_skewness = 2.0,
        .hotspot_ratio = 0.1,
    };

    QueryBenchmark benchmark(config);
    benchmark.run();

    return 0;
} 