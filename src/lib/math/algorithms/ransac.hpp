#pragma once

#include <algorithm>
#include <functional>
#include <random>
#include <type_traits>

#include "dlt.hpp"

// TODO sampling works by a random shuffle, then the samples are used in a serial manner (if serial sampling is enabled)
// a blue noise would be nice here (if it's possible at all)

// TODO finalize step for the solution calculation

template <typename E>
class RANSAC
{
 public:
    using ValueType = typename E::ValueType;
    using SolutionType = typename E::SolutionType;
    static_assert(std::is_base_of_v<EquationBuilder<ValueType>, E>,
                  "The template parameter of RANSAC must be an Equation builder");

    struct Settings
    {
        typename E::ConfigType builderConfig;
        unsigned int maxDataForConfiguration = 0;
        const unsigned int* seed = nullptr;
        bool serialSampling = true;
        bool validate() const { return maxDataForConfiguration > 0; }
        operator bool() const { return validate(); }
    };

    explicit RANSAC(const Settings& _settings)
      : settings(_settings),
        gen(settings.seed == nullptr ? rd() : *settings.seed),
        builder{settings.builderConfig} {
        ASSERT(settings.validate());
    }

    explicit RANSAC(Settings&& _settings)
      : settings(std::move(_settings)),
        gen(settings.seed == nullptr ? rd() : *settings.seed),
        builder{settings.builderConfig} {
        ASSERT(settings.validate());
    }

    struct DataBlock
    {
        MatrixRef<ValueType, RowMajorIndexing> data;
        std::function<void(E&, const ValueType*)> equationRegistry;
        // TODO adaptive threshold
        std::function<Float(const SolutionType&, const ValueType*)> acceptFunction;
        Float acceptLimit = Float{0};
        float matchWeight = 1;
        unsigned int sampleWeight = 1;
        unsigned int samplingCount = 0;
        unsigned int matchingCount = 0;
        struct Extra
        {
            unsigned int samplingOffset = 0;
            unsigned int matchingOffset = 0;
            mutable std::uniform_int_distribution<> sampleDis = std::uniform_int_distribution<>{};
            bool shuffled = false;
        } extra = {};
    };

    struct DataIndex
    {
        uint32_t blockId : 4;
        uint32_t entryId : 28;
    };

    void add(const DataBlock& block) {
        MEASURE("RANSAC::add&");
        ASSERT(block.data.rows() > 0);
        sumBlockSampleWeight += block.sampleWeight;
        updateBlockDist();
        dataBlocks.push_back(block);
        if (dataBlocks.back().samplingCount == 0)
            dataBlocks.back().samplingCount = dataBlocks.back().data.rows();
        if (dataBlocks.back().matchingCount == 0)
            dataBlocks.back().matchingCount = dataBlocks.back().data.rows();
    }

    void add(DataBlock&& block) {
        MEASURE("RANSAC::add&&");
        ASSERT(block.data.rows() > 0);
        sumBlockSampleWeight += block.sampleWeight;
        updateBlockDist();
        dataBlocks.push_back(std::move(block));
        if (dataBlocks.back().samplingCount == 0)
            dataBlocks.back().samplingCount = dataBlocks.back().data.rows();
        if (dataBlocks.back().matchingCount == 0)
            dataBlocks.back().matchingCount = dataBlocks.back().data.rows();
    }

    struct Configuration
    {
        SolutionType solution;
        float matchValue = 0;
        unsigned int matchCount = 0;
        friend bool operator<(const Configuration& lhs, const Configuration& rhs) {
            return lhs.matchValue < rhs.matchValue;
        }
#ifdef DEBUG
        std::vector<DataIndex> index;
#endif
    };

    struct ConfigurationSearchParams
    {
        unsigned int numTries = 0;
        bool validate() const { return numTries > 0; }
        operator bool() const { return validate(); }
    };

    void shuffleData() {
        MEASURE("RANSAC::shuffleData");
        for (DataBlock& block : dataBlocks) {
            std::shuffle(RowIterator<typename E::ValueType>{0, block.data},
                         RowIterator<typename E::ValueType>{block.data.rows(), block.data}, gen);
            block.extra.shuffled = true;
        }
    }

    bool findConfiguration(Configuration& config, const ConfigurationSearchParams& params) {
        MEASURE("RANSAC::findConfiguration");
        ASSERT(params.validate());
        E b{settings.builderConfig};
        INIT_ERROR(error(0.0));
        bool ret = false;
        for (unsigned int i = 0; i < params.numTries; ++i) {
            MEASURE("RANSAC::findConfiguration/iteration");
            initSampling();
            Configuration c;
            INIT_ERROR(error(c.solution));
            b.reset();
            unsigned int numData = 0;
            DataIndex s = sample();
            do {
                if (settings.serialSampling) {
                    s.entryId++;
                    if (s.entryId >= dataBlocks[s.blockId].extra.samplingOffset
                                       + dataBlocks[s.blockId].samplingCount
                        || s.entryId >= dataBlocks[s.blockId].data.rows())
                        s = sample();
                }
                else
                    s = sample();
#ifdef DEBUG
                ASSERT(!settings.serialSampling || dataBlocks[s.blockId].extra.shuffled);
#endif
                dataBlocks[s.blockId].equationRegistry(
                  b, &dataBlocks[s.blockId].data.at(s.entryId, 0u));
#ifdef DEBUG
                c.index.push_back(s);
#endif
            } while (++numData < settings.maxDataForConfiguration && !b.enough());
            ASSERT(b.enough());
            if (std::move(b).solveSimple(c.solution)) {
                REGISTER_ERROR(error(c.solution));
                auto m = match(c.solution);
                c.matchValue = m.first;
                c.matchCount = m.second;
                if (config < c) {
                    config = std::move(c);
                    ret = true;
                }
            }
        }
        REGISTER_ERROR(error(config.solution));
        return ret;
    }

    void configure(const Configuration& config) {
        MEASURE("RANSAC::configure&");
        configuration = config;
        initializeConfiguration();
    }
    void configure(Configuration&& config) {
        MEASURE("RANSAC::configure&&");
        configuration = std::move(config);
        initializeConfiguration();
    }

    bool configure(const ConfigurationSearchParams& params) {
        MEASURE("RANSAC::configure with params");
        Configuration config;
        bool ret = findConfiguration(config, params);
        if (ret)
            configure(std::move(config));
        return ret;
    }

    unsigned int match(const DataBlock* block, const SolutionType& solution) const {
        MEASURE("RANSAC::match block");
        unsigned int ret = 0;
        for (unsigned int r = 0;
             r < block->samplingCount && r + block->extra.samplingOffset < block->data.rows(); ++r)
            if (block->acceptFunction(solution,
                                      &block->data.at(r + block->extra.samplingOffset, 0u))
                <= acceptLimitScale * block->acceptLimit)
                ret++;
        return ret;
    }

    std::pair<float, unsigned int> match(const SolutionType& solution) const {
        MEASURE("RANSAC::match all");
        Float ret = 0;
        INIT_ERROR(error(ret));
        unsigned int count = 0;
        for (const DataBlock& block : dataBlocks) {
            unsigned int r = match(&block, solution);
            count += r;
            ret += static_cast<Float>(r) * Float{block.matchWeight};
        }
        REGISTER_ERROR(error(ret));
        return {static_cast<float>(ret), count};
    }

    const Configuration& getConfiguration() const { return configuration; }

    bool solveSimple(SolutionType& solution) const& {
        MEASURE("RANSAC::solveSimple&");
        return builder.solveSimple(solution);
    }

    bool solveSimple(SolutionType& solution) && {
        MEASURE("RANSAC::solveSimple&&");
        return std::move(builder).solveSimple(solution);
    }

 private:
    Settings settings;
    std::vector<DataBlock> dataBlocks;
    Configuration configuration;
    Float acceptLimitScale = Float{1};

    std::random_device rd;
    mutable std::mt19937 gen;
    mutable std::uniform_int_distribution<> blockDis;
    unsigned int sumBlockSampleWeight = 0;

    E builder;

    void initSampling() {
        MEASURE("RANSAC::initSampling");
        for (DataBlock& block : dataBlocks) {
            if (block.data.rows() <= block.samplingCount)
                block.extra.samplingOffset = 0;
            else {
                std::uniform_int_distribution<> dis{
                  0, static_cast<int>(block.data.rows() - block.samplingCount)};
                block.extra.samplingOffset = safe_cast<unsigned int>(dis(gen));
            }
            block.extra.sampleDis = std::uniform_int_distribution<>(
              safe_cast<int>(block.extra.samplingOffset),
              safe_cast<int>(
                std::min(block.extra.samplingOffset + block.samplingCount, block.data.rows()))
                - 1);
        }
    }

    unsigned int initMatching() {
        // TODO implement blue noise here istead
        MEASURE("RANSAC::initMatching");
        unsigned int count = 0;
        for (DataBlock& block : dataBlocks) {
            const unsigned int a =
              block.extra.samplingOffset + block.samplingCount <= block.matchingCount
                ? 0
                : block.extra.samplingOffset + block.samplingCount - block.matchingCount;
            const unsigned int b =
              block.extra.samplingOffset + block.matchingCount >= block.data.rows()
                ? block.data.rows()
                : block.extra.samplingOffset + block.matchingCount;
            if (b - a <= block.matchingCount)
                block.extra.matchingOffset = a;
            else {
                std::uniform_int_distribution<> dis{static_cast<int>(a),
                                                    static_cast<int>(b - block.matchingCount)};
                block.extra.matchingOffset = safe_cast<unsigned int>(dis(gen));
            }
            count += block.extra.matchingOffset + block.matchingCount <= block.data.rows()
                       ? block.matchingCount
                       : block.data.rows() - block.extra.matchingOffset;
        }
        return count;
    }

    void updateBlockDist() {
        ASSERT(sumBlockSampleWeight > 0);
        blockDis = std::uniform_int_distribution<>(0, safe_cast<int>(sumBlockSampleWeight) - 1);
    }

    DataIndex sample() const {
        MEASURE("RANSAC::sample");
        unsigned int blockSample = safe_cast<unsigned int>(blockDis(gen));
        DataIndex s;
        s.blockId = 0;
        s.entryId = 0;
        while (dataBlocks[s.blockId].sampleWeight <= blockSample) {
            blockSample -= dataBlocks[s.blockId].sampleWeight;
            s.blockId++;
        }
        s.entryId = safe_cast<unsigned int>(dataBlocks[s.blockId].extra.sampleDis(gen));
        return s;
    }

    void initializeConfiguration() {
        MEASURE("RANSAC::initializeConfiguration");
        const unsigned int count = initMatching();
        builder = E{settings.builderConfig, count};
        for (const DataBlock& block : dataBlocks) {
            for (unsigned int r = block.extra.matchingOffset;
                 r < block.data.rows() && r < block.extra.matchingOffset + block.matchingCount; ++r)
                if (block.acceptFunction(configuration.solution, &block.data.at(r, 0u))
                    <= acceptLimitScale * block.acceptLimit)
                    block.equationRegistry(builder, &block.data.at(r, 0u));
        }
    }
};
