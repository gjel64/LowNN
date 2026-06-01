#pragma once 
#include <random>

// singleton
class RandomGenerator {
private:

    std::mt19937 _rng{std::random_device{}()};
    static RandomGenerator* _instance;

    RandomGenerator() = default;
public:

    static RandomGenerator* getInstance();

    float rand_dist(float mean = 0.0f, float stddev = 1.0f);
};